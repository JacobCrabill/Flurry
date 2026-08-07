//! GPU compute back end: a Vulkan device, and the kernels the solver dispatches
//! onto it.
//!
//! The solver's operator applications are all one shape -- a dense row-major
//! `C = A*B` with `A` the reference-element operator and `B` the solution laid
//! out with the element index last -- which is exactly what spock's prebuilt
//! `dgemm` computes. So the first operators to move across need no kernel of
//! our own; `Mode.overwrite` and `.accumulate` are just `beta = 0` and `1`.
//!
//! Everything here is f64. Some GPUs support `shaderFloat64` poorly or not at
//! all, and this will refuse to run on those -- which is the right trade for a
//! solver whose accuracy claims are the point.
//!
//! State of the port: a whole time step, for the inviscid Euler equations in
//! either dimension -- the eight residual dispatches plus the Runge-Kutta
//! update, recorded into one command buffer and submitted once per stage.
//! Operands are device-resident, so a dispatch binds them where they lie rather
//! than copying them in and out; `gemmHost` still exists for callers holding
//! ordinary host memory.
//!
//! Three of the kernels have the Euler equations in them and so need `n_dims`
//! at compile time -- a runtime bound would leave the conserved state and the
//! metric terms dynamically indexed, which SPIR-V puts in private memory rather
//! than registers. `build.zig` compiles those three once per dimension from one
//! source, and `Kind.forDims` picks at dispatch. The rest -- the gemms, the
//! scatter and gather, the RK update -- carry no equation and no dimension.
//!
//! What is left on the CPU: the residual norms and the error measure, both of
//! which run at report intervals rather than every step, and anything the
//! kernels do not cover -- advection-diffusion, the viscous terms, and the
//! viscous wall conditions -- each of which falls back rather than failing.
//!
//! Because nothing on the CPU reads the arrays between reports, they live in
//! the device's own memory rather than host-visible memory, which is worth an
//! order of magnitude: the same dgemm measured 1.32 GB/s on one and 15.09 GB/s
//! on the other, and 25.5 GB/s once spock's dgemm was tiled over rows. A case
//! that does fall back keeps host-visible arrays, since it would otherwise read
//! a stale block.

const std = @import("std");
const spock = @import("spock");

const dgemm = spock.kernels.blas.dgemm;
const dgemm_spv = @embedFile("spock/dgemm.spv");

const flux_euler = @import("kernels/flux_euler.zig");

pub const face_scatter = @import("kernels/face_scatter.zig");
pub const face_scatter_grad = @import("kernels/face_scatter_grad.zig");
pub const face_gather = @import("kernels/face_gather.zig");
pub const face_common_u = @import("kernels/face_common_u.zig");
pub const face_common_f = @import("kernels/face_common_f.zig");
pub const face_bcs = @import("kernels/face_bcs.zig");
pub const face_bcs_grad = @import("kernels/face_bcs_grad.zig");
pub const rk_update = @import("kernels/rk_update.zig");

const face_scatter_spv = @embedFile("face_scatter.spv");
const face_scatter_grad_spv = @embedFile("face_scatter_grad.spv");
const face_gather_spv = @embedFile("face_gather.spv");
const face_common_u_spv = @embedFile("face_common_u.spv");
const face_bcs_grad_spv = @embedFile("face_bcs_grad.spv");
const rk_update_spv = @embedFile("rk_update.spv");

// The kernels with the equations in them are built once per dimension; see the
// kernel loop in `build.zig`. Each file's host-facing half -- its push-constant
// layout and workgroup size -- is the same for both, which is why there is still
// only one import of each above.
const face_bcs_spv = [2][]const u8{ @embedFile("face_bcs_2d.spv"), @embedFile("face_bcs_3d.spv") };

// The two flux kernels carry a second axis as well: viscous binds the gradient
// arrays and adds the stress tensor, which the inviscid build has no use for.
const flux_euler_spv = [2][2][]const u8{
    .{ @embedFile("flux_euler_2d.spv"), @embedFile("flux_euler_2d_visc.spv") },
    .{ @embedFile("flux_euler_3d.spv"), @embedFile("flux_euler_3d_visc.spv") },
};
const face_common_f_spv = [2][2][]const u8{
    .{ @embedFile("face_common_f_2d.spv"), @embedFile("face_common_f_2d_visc.spv") },
    .{ @embedFile("face_common_f_3d.spv"), @embedFile("face_common_f_3d_visc.spv") },
};

/// What the index arrays use for "no global flux point here". The host's
/// `geo.none` is `maxInt(usize)`; it narrows to this on upload.
pub const none_u32 = face_scatter.none;

pub const Error = error{
    /// No Vulkan loader, or no device on it with compute support
    NoComputeDevice,
    /// A batch tried to record the same kernel twice. Each kernel owns one
    /// descriptor set, so the second recording would overwrite the first's
    /// arguments and both dispatches would run with the second's -- silently.
    KernelAlreadyRecorded,
    /// The device rejected something, or a dispatch failed. The underlying
    /// Vulkan error is reported to stderr -- it is not in this set because
    /// hoisting the whole of `vk.Error` into every caller's signature buys
    /// nothing they could act on.
    DeviceFailure,
    OutOfMemory,
};

/// The kernels a `Device` owns. A batch may record each at most once.
///
/// The three with the Euler equations in them appear twice, once per dimension:
/// a run uses one or the other, but a `Device` is built before the case is and
/// the tests share one across both, so both are compiled in. The cost is three
/// extra pipelines at startup.
const Kind = enum {
    dgemm,
    flux_euler_2d,
    flux_euler_3d,
    flux_euler_2d_visc,
    flux_euler_3d_visc,
    face_scatter,
    face_scatter_grad,
    face_gather,
    face_common_u,
    face_common_f_2d,
    face_common_f_3d,
    face_common_f_2d_visc,
    face_common_f_3d_visc,
    face_bcs_2d,
    face_bcs_3d,
    face_bcs_grad,
    rk_update,

    /// The variant of `kind` for a case of `n_dims` dimensions, viscous or not.
    /// `kind` must be the 2D inviscid one, which is how the tables below are
    /// keyed.
    fn forCase(kind: Kind, n_dims: usize, visc: bool) Kind {
        const three = n_dims == 3;
        return switch (kind) {
            .flux_euler_2d => switch (visc) {
                false => if (three) .flux_euler_3d else .flux_euler_2d,
                true => if (three) .flux_euler_3d_visc else .flux_euler_2d_visc,
            },
            .face_common_f_2d => switch (visc) {
                false => if (three) .face_common_f_3d else .face_common_f_2d,
                true => if (three) .face_common_f_3d_visc else .face_common_f_2d_visc,
            },
            .face_bcs_2d => if (three) .face_bcs_3d else .face_bcs_2d,
            else => kind,
        };
    }
};

/// How many times one batch may dispatch each kernel.
///
/// A kernel owns a single descriptor set, so a second recording would leave both
/// dispatches using the second set of arguments -- no error, a plausible wrong
/// answer. Each repeat therefore needs its own instance, and these are the
/// counts a whole step actually uses: three products (`extrapolateU` and the two
/// divergences) and two RK updates (saving the solution, then advancing it).
const pool_sizes = std.enums.EnumArray(Kind, u8).init(.{
    // A viscous residual runs six products, not three: the gradient at the
    // solution points and its correction, then the two divergences, plus
    // `extrapolateU` and one `extrapolateGrad` per dimension.
    .dgemm = 8,
    .flux_euler_2d = 1,
    .flux_euler_3d = 1,
    .flux_euler_2d_visc = 1,
    .flux_euler_3d_visc = 1,
    // Twice per viscous residual: the solution, then `u_ldg` beside it
    .face_scatter = 2,
    .face_scatter_grad = 1,
    // Twice per viscous residual: the common solution, then the common flux
    .face_gather = 2,
    .face_common_u = 1,
    .face_common_f_2d = 1,
    .face_common_f_3d = 1,
    .face_common_f_2d_visc = 1,
    .face_common_f_3d_visc = 1,
    .face_bcs_2d = 1,
    .face_bcs_3d = 1,
    .face_bcs_grad = 1,
    .rk_update = 2,
});

/// How many times one batch may record a dgemm. For tests: the count is a
/// property of the longest residual, not something a caller should hard-code.
pub fn dgemmPoolSizeForTest() usize {
    return pool_sizes.get(.dgemm);
}

/// Instances of one kernel, handed out one per recording in a batch.
const Pool = struct {
    instances: []spock.Kernel,
    used: usize = 0,

    fn next(p: *Pool, batching: bool) Error!*spock.Kernel {
        if (!batching) return &p.instances[0];
        if (p.used >= p.instances.len) return error.KernelAlreadyRecorded;
        defer p.used += 1;
        return &p.instances[p.used];
    }
};

/// A device buffer handle, as a dispatch binds it.
pub const Buffer = spock.vk.Buffer;

/// One storage-buffer argument: a whole buffer, or a byte range of one.
/// `.whole(buf)` is the common case.
pub const Binding = spock.Kernel.Binding;

/// Where a `Heap`'s allocations live: host-visible, or the device's own memory.
pub const Location = spock.Location;

/// Whether a product overwrites its destination or accumulates into it,
/// matching the CPU `gemm`'s `Mode`.
pub const Mode = enum { overwrite, accumulate };

/// Which GPU to take when there is more than one.
///
/// Mirrors spock's own `Pick`, which it does not export -- `Context.Options` is
/// only reachable as an inferred anonymous literal.
pub const Pick = enum { first, largest };

pub const Options = struct {
    pick: Pick = .largest,
    /// Turn on the Vulkan validation layer, if it is installed
    validate: bool = false,
};

/// A compute device plus the kernels compiled against it.
///
/// The `Context` is on the heap rather than inline. Every spock `Kernel` and
/// `Buffer` keeps a `*Context`, so an inline one would make a `Device` unsafe to
/// move -- copy it out of the local it was built in and every kernel is left
/// pointing at a dead stack frame, which shows up as a null function pointer at
/// the first dispatch rather than anywhere near the mistake. One allocation
/// removes the hazard instead of documenting it.
pub const Device = struct {
    gpa: std.mem.Allocator,
    ctx: *spock.Context,

    /// Every kernel, with its instances; see `pool_sizes`.
    pools: std.enums.EnumArray(Kind, Pool),
    /// One slab behind every pool's `instances`.
    all_kernels: []spock.Kernel,

    /// Reused across steps; `beginBatch` resets it rather than allocating a new
    /// command buffer every residual.
    batch: spock.Pipeline,
    /// Whether a batch is open, so dispatches are recorded rather than run.
    batching: bool = false,

    /// Staging for `gemmHost`, grown on demand and reused across calls.
    stage_a: Scratch = .{},
    stage_b: Scratch = .{},
    stage_c: Scratch = .{},

    pub fn init(gpa: std.mem.Allocator, opts: Options) Error!Device {
        const ctx = try gpa.create(spock.Context);
        errdefer gpa.destroy(ctx);

        ctx.* = spock.Context.init(gpa, .{
            .pick = switch (opts.pick) {
                .first => .first,
                .largest => .largest,
            },
            .validate = opts.validate,
            .app_name = "flurry",
        }) catch |err| return translate(err, "creating the compute context");
        errdefer ctx.deinit();

        // Every kernel, with its instances, out of one slab.
        const specs = std.enums.EnumArray(Kind, Spec).init(.{
            // spock's dgemm names its entry point after itself, not "main"
            .dgemm = .{ dgemm_spv, "dgemm", 3, @sizeOf(dgemm.PushConstants) },
            .flux_euler_2d = .{ flux_euler_spv[0][0], "flux_euler", 3, @sizeOf(flux_euler.PushConstants) },
            .flux_euler_3d = .{ flux_euler_spv[1][0], "flux_euler", 3, @sizeOf(flux_euler.PushConstants) },
            .flux_euler_2d_visc = .{ flux_euler_spv[0][1], "flux_euler", 5, @sizeOf(flux_euler.PushConstants) },
            .flux_euler_3d_visc = .{ flux_euler_spv[1][1], "flux_euler", 5, @sizeOf(flux_euler.PushConstants) },
            .face_scatter = .{ face_scatter_spv, "face_scatter", 4, @sizeOf(face_scatter.PushConstants) },
            .face_scatter_grad = .{ face_scatter_grad_spv, "face_scatter_grad", 4, @sizeOf(face_scatter_grad.PushConstants) },
            .face_gather = .{ face_gather_spv, "face_gather", 4, @sizeOf(face_gather.PushConstants) },
            .face_common_u = .{ face_common_u_spv, "face_common_u", 2, @sizeOf(face_common_u.PushConstants) },
            .face_common_f_2d = .{ face_common_f_spv[0][0], "face_common_f", 5, @sizeOf(face_common_f.PushConstants) },
            .face_common_f_3d = .{ face_common_f_spv[1][0], "face_common_f", 5, @sizeOf(face_common_f.PushConstants) },
            .face_common_f_2d_visc = .{ face_common_f_spv[0][1], "face_common_f", 7, @sizeOf(face_common_f.PushConstants) },
            .face_common_f_3d_visc = .{ face_common_f_spv[1][1], "face_common_f", 7, @sizeOf(face_common_f.PushConstants) },
            .face_bcs_2d = .{ face_bcs_spv[0], "face_bcs", 4, @sizeOf(face_bcs.PushConstants) },
            .face_bcs_3d = .{ face_bcs_spv[1], "face_bcs", 4, @sizeOf(face_bcs.PushConstants) },
            .face_bcs_grad = .{ face_bcs_grad_spv, "face_bcs_grad", 1, @sizeOf(face_bcs_grad.PushConstants) },
            .rk_update = .{ rk_update_spv, "rk_update", 4, @sizeOf(rk_update.PushConstants) },
        });

        var total: usize = 0;
        for (pool_sizes.values) |n| total += n;

        const slab = try gpa.alloc(spock.Kernel, total);
        errdefer gpa.free(slab);

        var built: usize = 0;
        errdefer for (slab[0..built]) |*k| k.deinit();

        var pools: std.enums.EnumArray(Kind, Pool) = undefined;
        for (std.enums.values(Kind)) |kind| {
            const spec = specs.get(kind);
            const n = pool_sizes.get(kind);
            const start = built;
            for (0..n) |_| {
                slab[built] = try makeKernel(ctx, spec[0], spec[1], spec[2], spec[3]);
                built += 1;
            }
            pools.set(kind, .{ .instances = slab[start..built] });
        }

        const batch = spock.Pipeline.create(ctx) catch |err| {
            return translate(err, "creating the dispatch batch");
        };

        return .{
            .gpa = gpa,
            .ctx = ctx,
            .pools = pools,
            .all_kernels = slab,
            .batch = batch,
        };
    }

    pub fn deinit(d: *Device) void {
        d.stage_a.deinit();
        d.stage_b.deinit();
        d.stage_c.deinit();
        d.batch.deinit();
        for (d.all_kernels) |*k| k.deinit();
        d.gpa.free(d.all_kernels);
        d.ctx.deinit();
        d.gpa.destroy(d.ctx);
    }

    pub fn name(d: *const Device) []const u8 {
        return d.ctx.deviceName();
    }

    // ---- Batching ----

    /// Begin a run of dispatches to submit together.
    ///
    /// Each dispatch on its own costs a submit and a fence wait, and for
    /// matrices this small that round trip dwarfs the arithmetic. Recorded into
    /// one command buffer, a run of consecutive GPU work costs one round trip
    /// instead of one each, with a storage barrier between them so each sees the
    /// last one's writes.
    ///
    /// Only *consecutive* GPU work can go in one batch: anything the CPU has to
    /// touch in between ends it.
    pub fn beginBatch(d: *Device) Error!void {
        std.debug.assert(!d.batching); // a batch is already open
        d.batch.reset() catch |err| return translate(err, "resetting the batch");
        for (&d.pools.values) |*p| p.used = 0;
        d.batching = true;
    }

    /// Abandon the open batch without submitting it.
    ///
    /// For the error path: a dispatch that fails part-way through recording
    /// would otherwise leave the batch open, and every later `beginBatch` would
    /// trip over it. The command buffer is reset by the next `beginBatch`, so
    /// there is nothing to undo here beyond the flag.
    pub fn abortBatch(d: *Device) void {
        d.batching = false;
    }

    /// Whether a batch is open, for a caller deciding whether to open its own.
    pub fn isBatching(d: *const Device) bool {
        return d.batching;
    }

    /// Submit everything recorded since `beginBatch` and wait for it.
    pub fn submitBatch(d: *Device) Error!void {
        std.debug.assert(d.batching); // no batch is open
        d.batching = false;
        d.batch.submit() catch |err| return translate(err, "submitting the batch");
        d.ctx.wait() catch |err| return translate(err, "waiting on the batch");
    }

    /// Record a dispatch into the open batch, or run it on its own if none is.
    ///
    /// Takes a fresh instance of the kernel each time within a batch, and fails
    /// with `KernelAlreadyRecorded` when the pool runs out -- reusing one would
    /// leave both dispatches running with the second set of arguments.
    fn run(
        d: *Device,
        kind: Kind,
        args: spock.Kernel.DispatchArgs,
        what: []const u8,
    ) Error!void {
        const kernel = try d.pools.getPtr(kind).next(d.batching);

        if (d.batching) {
            d.batch.addKernel(kernel, args) catch |err| return translate(err, what);
            return;
        }

        kernel.dispatch(args) catch |err| return translate(err, what);
        // `dispatch` submits without blocking; the results are not there until
        // the fence clears.
        d.ctx.wait() catch |err| return translate(err, what);
    }

    /// `C = A*B`, or `C += A*B`, over buffers that are already on the device:
    /// the GPU counterpart of the solver's `gemm`, with the same argument order.
    ///
    /// `A` is (m, k), `B` is (k, n), `C` is (m, n), all row-major and densely
    /// packed. Nothing is copied -- this is what allocating the solver's arrays
    /// through `Heap` buys.
    pub fn gemm(
        d: *Device,
        m: usize,
        n: usize,
        k: usize,
        a: Binding,
        b: Binding,
        c: Binding,
        mode: Mode,
    ) Error!void {
        if (m == 0 or n == 0 or k == 0) return;

        const pc: dgemm.PushConstants = .{
            .M = @intCast(m),
            .K = @intCast(k),
            .N = @intCast(n),
            .alpha = 1.0,
            .beta = if (mode == .accumulate) 1.0 else 0.0,
        };

        try d.run(.dgemm, .{
            .buffers = &.{ a, b, c },
            .push_constant = std.mem.asBytes(&pc),
            // The kernel tiles over rows and dispatches in two dimensions, so
            // the mapping comes from it rather than from here.
            .groups = dgemm.groups(pc.M, pc.N),
        }, "dispatching dgemm");
    }

    /// Euler flux at the solution points, in reference space, with the viscous
    /// terms when `visc` is given.
    ///
    /// One thread per `(spt, ele)`; see `kernels/flux_euler.zig` for the layouts
    /// it assumes.
    pub fn fluxEuler(
        d: *Device,
        n_dims: usize,
        n_spts: usize,
        n_eles: usize,
        pc: flux_euler.PushConstants,
        u_spts: Binding,
        inv_jaco: Binding,
        f_spts: Binding,
        /// Both present for a viscous case, both absent otherwise -- they are
        /// the two bindings the viscous build adds.
        visc: ?struct { du_spts: Binding, jaco_det: Binding },
    ) Error!void {
        if (n_spts == 0 or n_eles == 0) return;

        const threads: u32 = @intCast(n_spts * n_eles);
        const groups = std.math.divCeil(u32, threads, flux_euler.WgSize.x) catch unreachable;
        const kind = Kind.forCase(.flux_euler_2d, n_dims, visc != null);

        if (visc) |v| {
            return d.run(kind, .{
                .buffers = &.{ u_spts, inv_jaco, f_spts, v.du_spts, v.jaco_det },
                .push_constant = std.mem.asBytes(&pc),
                .groups = .{ groups, 1, 1 },
            }, "dispatching flux_euler");
        }
        try d.run(kind, .{
            .buffers = &.{ u_spts, inv_jaco, f_spts },
            .push_constant = std.mem.asBytes(&pc),
            .groups = .{ groups, 1, 1 },
        }, "dispatching flux_euler");
    }

    /// Each element's flux-point gradient -> the shared face arrays.
    pub fn faceScatterGrad(
        d: *Device,
        p: face_scatter_grad.PushConstants,
        du_fpts: Binding,
        fpt2gfpt: Binding,
        fpt2slot: Binding,
        faces_du: Binding,
    ) Error!void {
        if (p.n_fpts == 0 or p.n_eles == 0) return;
        try d.run(.face_scatter_grad, .{
            .buffers = &.{ du_fpts, fpt2gfpt, fpt2slot, faces_du },
            .push_constant = std.mem.asBytes(&p),
            .groups = .{ groupsFor(p.n_fpts * p.n_eles, face_scatter_grad.WgSize.x), 1, 1 },
        }, "dispatching face_scatter_grad");
    }

    /// Single-valued interface solution, for the viscous gradient correction.
    pub fn faceCommonU(
        d: *Device,
        p: face_common_u.PushConstants,
        u_ldg: Binding,
        u_comm: Binding,
    ) Error!void {
        if (p.n_gfpts == 0) return;
        try d.run(.face_common_u, .{
            .buffers = &.{ u_ldg, u_comm },
            .push_constant = std.mem.asBytes(&p),
            .groups = .{ groupsFor(p.n_gfpts, face_common_u.WgSize.x), 1, 1 },
        }, "dispatching face_common_u");
    }

    /// Boundary gradients: every condition with a kernel extrapolates.
    pub fn faceBcsGrad(d: *Device, p: face_bcs_grad.PushConstants, du: Binding) Error!void {
        if (p.n_gfpts_bnd == 0) return;
        try d.run(.face_bcs_grad, .{
            .buffers = &.{du},
            .push_constant = std.mem.asBytes(&p),
            .groups = .{ groupsFor(p.n_gfpts_bnd, face_bcs_grad.WgSize.x), 1, 1 },
        }, "dispatching face_bcs_grad");
    }

    // ---- The face path ----

    /// Each element's flux-point solution -> the shared face arrays.
    pub fn faceScatter(
        d: *Device,
        n_fpts: usize,
        n_eles: usize,
        n_vars: usize,
        n_gfpts: usize,
        u_fpts: Binding,
        fpt2gfpt: Binding,
        fpt2slot: Binding,
        faces_u: Binding,
    ) Error!void {
        if (n_fpts == 0 or n_eles == 0) return;
        const p: face_scatter.PushConstants = .{
            .n_fpts = @intCast(n_fpts),
            .n_eles = @intCast(n_eles),
            .n_vars = @intCast(n_vars),
            .n_gfpts = @intCast(n_gfpts),
        };
        try d.run(.face_scatter, .{
            .buffers = &.{ u_fpts, fpt2gfpt, fpt2slot, faces_u },
            .push_constant = std.mem.asBytes(&p),
            .groups = .{ groupsFor(n_fpts * n_eles, face_scatter.WgSize.x), 1, 1 },
        }, "dispatching face_scatter");
    }

    /// The common normal flux -> each element's own flux-point array.
    pub fn faceGather(
        d: *Device,
        n_fpts: usize,
        n_eles: usize,
        n_vars: usize,
        n_gfpts: usize,
        faces_f: Binding,
        fpt2gfpt: Binding,
        fpt2slot: Binding,
        f_comm: Binding,
    ) Error!void {
        if (n_fpts == 0 or n_eles == 0) return;
        const p: face_gather.PushConstants = .{
            .n_fpts = @intCast(n_fpts),
            .n_eles = @intCast(n_eles),
            .n_vars = @intCast(n_vars),
            .n_gfpts = @intCast(n_gfpts),
        };
        try d.run(.face_gather, .{
            .buffers = &.{ faces_f, fpt2gfpt, fpt2slot, f_comm },
            .push_constant = std.mem.asBytes(&p),
            .groups = .{ groupsFor(n_fpts * n_eles, face_gather.WgSize.x), 1, 1 },
        }, "dispatching face_gather");
    }

    /// Rusanov common normal flux at every global flux point.
    pub fn faceCommonF(
        d: *Device,
        n_dims: usize,
        p: face_common_f.PushConstants,
        u: Binding,
        norm: Binding,
        d_a: Binding,
        f_comm: Binding,
        wave_sp: Binding,
        /// The two bindings the viscous build adds; null for an inviscid case.
        visc: ?struct { u_ldg: Binding, du: Binding },
    ) Error!void {
        if (p.n_gfpts == 0) return;
        const kind = Kind.forCase(.face_common_f_2d, n_dims, visc != null);
        const groups = groupsFor(p.n_gfpts, face_common_f.WgSize.x);

        if (visc) |v| {
            return d.run(kind, .{
                .buffers = &.{ u, norm, d_a, f_comm, wave_sp, v.u_ldg, v.du },
                .push_constant = std.mem.asBytes(&p),
                .groups = .{ groups, 1, 1 },
            }, "dispatching face_common_f");
        }
        try d.run(kind, .{
            .buffers = &.{ u, norm, d_a, f_comm, wave_sp },
            .push_constant = std.mem.asBytes(&p),
            .groups = .{ groups, 1, 1 },
        }, "dispatching face_common_f");
    }

    /// Ghost states at the boundary flux points.
    pub fn faceBcs(
        d: *Device,
        n_dims: usize,
        p: face_bcs.PushConstants,
        u: Binding,
        norm: Binding,
        bc_code: Binding,
        /// The prescribed viscous state. An inviscid case has no such array and
        /// passes `u`, making the kernel's second write a no-op.
        u_ldg: Binding,
    ) Error!void {
        if (p.n_gfpts_bnd == 0) return;
        try d.run(Kind.forCase(.face_bcs_2d, n_dims, false), .{
            .buffers = &.{ u, norm, bc_code, u_ldg },
            .push_constant = std.mem.asBytes(&p),
            .groups = .{ groupsFor(p.n_gfpts_bnd, face_bcs.WgSize.x), 1, 1 },
        }, "dispatching face_bcs");
    }

    /// `dst = src - dt/|J| * sum_t coeff[t] * divf[first_stage + t]`.
    ///
    /// With no terms it is a copy, which is how the time loop saves the solution
    /// at the start of a step.
    pub fn rkUpdate(
        d: *Device,
        p: rk_update.PushConstants,
        src: Binding,
        divf: Binding,
        jaco_det: Binding,
        dst: Binding,
    ) Error!void {
        const total = p.n_spts * p.n_vars * p.n_eles;
        if (total == 0) return;
        try d.run(.rk_update, .{
            .buffers = &.{ src, divf, jaco_det, dst },
            .push_constant = std.mem.asBytes(&p),
            .groups = .{ groupsFor(total, rk_update.WgSize.x), 1, 1 },
        }, "dispatching rk_update");
    }

    /// The same product over ordinary host slices, staged in and out around the
    /// dispatch.
    ///
    /// Only for callers whose data is not already device-resident; the solver's
    /// own arrays are, and use `gemm`. The staging buffers are grown on demand
    /// and reused.
    pub fn gemmHost(
        d: *Device,
        m: usize,
        n: usize,
        k: usize,
        a: []const f64,
        b: []const f64,
        c: []f64,
        mode: Mode,
    ) Error!void {
        std.debug.assert(a.len >= m * k);
        std.debug.assert(b.len >= k * n);
        std.debug.assert(c.len >= m * n);
        if (m == 0 or n == 0 or k == 0) return;

        const a_buf = try d.stage_a.ensure(d.ctx, m * k);
        const b_buf = try d.stage_b.ensure(d.ctx, k * n);
        const c_buf = try d.stage_c.ensure(d.ctx, m * n);

        a_buf.copyFromHost(a[0 .. m * k]);
        b_buf.copyFromHost(b[0 .. k * n]);
        // With beta = 0 the kernel writes every element, so there is nothing
        // worth uploading.
        if (mode == .accumulate) c_buf.copyFromHost(c[0 .. m * n]);

        try d.gemm(m, n, k, .whole(a_buf.raw()), .whole(b_buf.raw()), .whole(c_buf.raw()), mode);

        c_buf.copyToHost(c[0 .. m * n]);
    }
};

fn makeKernel(
    ctx: *spock.Context,
    spirv: []const u8,
    entry: [*:0]const u8,
    buffers: u32,
    pc_size: u32,
) Error!spock.Kernel {
    return spock.Kernel.create(ctx, .{
        .spirv = spirv,
        .entry = entry,
        .buffers = buffers,
        .push_constant_size = pc_size,
    }) catch |err| translate(err, "building a compute pipeline");
}

fn groupsFor(threads: usize, wg: u32) u32 {
    return std.math.divCeil(u32, @intCast(threads), wg) catch unreachable;
}

/// What one kernel is built from: module, entry point, buffer count, push size.
const Spec = struct { []const u8, [*:0]const u8, u32, u32 };

/// A device buffer that grows to fit whatever it is asked to hold.
const Scratch = struct {
    buf: ?spock.Buffer(f64) = null,

    fn ensure(s: *Scratch, ctx: *spock.Context, n: usize) Error!spock.Buffer(f64) {
        if (s.buf) |b| {
            if (b.len >= n) return b;
            b.deinit();
            s.buf = null;
        }
        s.buf = spock.Buffer(f64).create(ctx, n) catch |err| {
            return translate(err, "allocating a device buffer");
        };
        return s.buf.?;
    }

    fn deinit(s: *Scratch) void {
        if (s.buf) |b| b.deinit();
        s.buf = null;
    }
};

/// Collapse a Vulkan error into this module's set, reporting what was lost.
fn translate(err: anyerror, doing: []const u8) Error {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.NoComputeDevice, error.InitializationFailed => error.NoComputeDevice,
        else => {
            std.debug.print("gpu: {t} while {s}\n", .{ err, doing });
            return error.DeviceFailure;
        },
    };
}

/// A `std.mem.Allocator` for the arrays a dispatch binds.
///
/// Two modes, chosen by `Location`:
///
///   * `.host` -- one host-visible, persistently mapped buffer per allocation.
///     What comes back *is* the device memory, so operations still on the CPU
///     keep reading and writing it as an ordinary slice and nothing has to be
///     synchronised. That is what let the port proceed one operator at a time.
///
///   * `.device` -- a pair: a host-visible block, which is what the allocation
///     returns, and a device-local one the kernels actually bind. They are
///     separate memory, so the two have to be pushed together explicitly with
///     `syncToDevice` / `syncToHost`. Only worth it once nothing on the CPU
///     touches these arrays between those points.
///
/// The second is much faster and the first is much simpler. Measured on a
/// Quadro T1000, a dgemm of the shape this solver dispatches ran at 1.30 GB/s
/// on host-visible memory -- an order of magnitude under PCIe, two under the
/// card's own memory.
///
/// Every allocation is its own Vulkan buffer, because spock binds descriptors
/// with `offset = 0, range = WHOLE_SIZE` and gives no way to point at part of
/// one. That makes this the wrong tool for many small allocations and the right
/// one for the handful of large solver arrays it holds.
pub const Heap = struct {
    dev: *Device,
    /// For the bookkeeping list only; the blocks themselves are device memory
    gpa: std.mem.Allocator,
    location: spock.Location,
    blocks: std.ArrayList(Block),

    const Block = struct {
        ptr: [*]u8,
        len: usize,
        /// Host-visible, and mapped to `ptr`
        host: spock.Buffer(u8),
        /// What a dispatch binds. The same as `host` in `.host` mode.
        device: spock.Buffer(u8),
    };

    pub fn init(dev: *Device, gpa: std.mem.Allocator, location: spock.Location) Heap {
        return .{ .dev = dev, .gpa = gpa, .location = location, .blocks = .empty };
    }

    pub fn deinit(h: *Heap) void {
        for (h.blocks.items) |b| {
            if (h.location == .device) b.device.deinit();
            b.host.deinit();
        }
        h.blocks.deinit(h.gpa);
    }

    pub fn allocator(h: *Heap) std.mem.Allocator {
        return .{
            .ptr = h,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .remap = remap,
                .free = free,
            },
        };
    }

    /// The buffer a dispatch should bind for `ptr`, or null if it did not come
    /// from here.
    ///
    /// A linear scan over a handful of blocks, and only when a dispatch is being
    /// recorded, so there is nothing to gain from an index.
    pub fn bufferFor(h: *const Heap, ptr: *const anyopaque) ?spock.vk.Buffer {
        const b = h.blockFor(ptr) orelse return null;
        return b.device.raw();
    }

    fn blockFor(h: *const Heap, ptr: *const anyopaque) ?Block {
        const addr = @intFromPtr(ptr);
        for (h.blocks.items) |b| {
            const base = @intFromPtr(b.ptr);
            if (addr >= base and addr < base + b.len) return b;
        }
        return null;
    }

    /// Push every block's host contents to the device.
    ///
    /// A no-op in `.host` mode, where there is only one copy of anything. Coarse
    /// on purpose: this runs when the host has just written the arrays, which is
    /// setup and the initial condition, not per step.
    pub fn syncToDevice(h: *Heap) Error!void {
        if (h.location == .host) return;
        for (h.blocks.items) |b| {
            b.device.copyFrom(b.host) catch |err| return translate(err, "uploading to the device");
        }
    }

    /// Pull every block's contents back, for the host to read.
    ///
    /// Called before the residual norms, the error measure and solution output
    /// -- all of which run at report intervals rather than every step.
    pub fn syncToHost(h: *Heap) Error!void {
        if (h.location == .host) return;
        for (h.blocks.items) |b| {
            b.host.copyFrom(b.device) catch |err| return translate(err, "reading back from the device");
        }
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, _: usize) ?[*]u8 {
        const h: *Heap = @ptrCast(@alignCast(ctx));

        h.blocks.ensureUnusedCapacity(h.gpa, 1) catch return null;
        const host = spock.Buffer(u8).create(h.dev.ctx, len) catch return null;

        // A mapped Vulkan allocation starts at the base of its own memory
        // object, so it is aligned far past anything an f64 array asks for --
        // but say so rather than assume it.
        const ptr = @as([*]u8, @ptrCast(host.ptr.?));
        if (!alignment.check(@intFromPtr(ptr))) {
            host.deinit();
            return null;
        }

        const device = switch (h.location) {
            .host => host,
            .device => spock.Buffer(u8).createIn(h.dev.ctx, len, .device) catch {
                host.deinit();
                return null;
            },
        };

        h.blocks.appendAssumeCapacity(.{ .ptr = ptr, .len = len, .host = host, .device = device });
        return ptr;
    }

    /// Never in place: a block is a whole Vulkan buffer, and growing one means
    /// a new allocation and a new descriptor. The solver never resizes these.
    fn resize(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) bool {
        return false;
    }

    fn remap(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) ?[*]u8 {
        return null;
    }

    fn free(ctx: *anyopaque, memory: []u8, _: std.mem.Alignment, _: usize) void {
        const h: *Heap = @ptrCast(@alignCast(ctx));
        for (h.blocks.items, 0..) |b, i| {
            if (b.ptr != memory.ptr) continue;
            if (h.location == .device) b.device.deinit();
            b.host.deinit();
            _ = h.blocks.swapRemove(i);
            return;
        }
        unreachable; // freed something this heap never handed out
    }
};

/// A standalone device buffer holding a copy of host data.
///
/// For the reference-element operators: they are constant for the whole run and
/// tiny -- an order-3 quad's `oppE` is 16x16 -- so uploading a copy at setup is
/// cheaper than routing `Element`'s allocations through a `Heap` would be
/// invasive.
pub const Array = struct {
    buf: spock.Buffer(f64),

    pub fn upload(d: *Device, src: []const f64) Error!Array {
        const buf = spock.Buffer(f64).create(d.ctx, src.len) catch |err| {
            return translate(err, "allocating a device array");
        };
        buf.copyFromHost(src);
        return .{ .buf = buf };
    }

    pub fn raw(a: Array) spock.vk.Buffer {
        return a.buf.raw();
    }

    pub fn binding(a: Array) Binding {
        return .whole(a.buf.raw());
    }

    pub fn deinit(a: *Array) void {
        a.buf.deinit();
    }
};

/// The integer counterpart of `Array`, for the connectivity a face kernel
/// walks: `fpt2gfpt`, its slot table, and the per-point boundary codes.
pub const IndexArray = struct {
    buf: spock.Buffer(u32),

    pub fn upload(d: *Device, src: []const u32) Error!IndexArray {
        const buf = spock.Buffer(u32).create(d.ctx, @max(src.len, 1)) catch |err| {
            return translate(err, "allocating a device index array");
        };
        buf.copyFromHost(src);
        return .{ .buf = buf };
    }

    pub fn binding(a: IndexArray) Binding {
        return .whole(a.buf.raw());
    }

    pub fn deinit(a: *IndexArray) void {
        a.buf.deinit();
    }
};

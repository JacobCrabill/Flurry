//! Boundary states at the global flux points.
//!
//! The GPU half of `Faces.applyBcsDim`, for the inviscid Euler equations. One
//! thread per boundary flux point: it reads the interior side and writes the
//! ghost state the Riemann solver will see opposite it.
//!
//! Each point carries its own condition as a `Code`, resolved on the host from
//! `gfpt2bnd` and `bc_list` once at setup -- the kernel does not need either
//! indirection, only the answer. Conditions outside this set (the viscous walls)
//! keep the whole step on the CPU; see `Solver.applyFaceBcs`.
//!
//! Layouts:
//!
//!     u       (slot, var, gfpt)   gfpt fastest; slot 0 in, slot 1 out
//!     u_ldg   (slot, var, gfpt)   the state prescribed to the viscous flux
//!     norm    (dim, gfpt)
//!     bc_code (gfpt - n_gfpts_int)
//!
//! Every condition here prescribes the same state to both, so `u_ldg` gets the
//! same ghost state `u` does. They only diverge at a no-slip wall, which has no
//! code and keeps the whole step on the CPU. An inviscid run has no `u_ldg`
//! array at all and binds `u` in its place, making the second write a no-op.
//!
//! Built once per dimension; see `n_dims` below and the kernel loop in
//! `build.zig`.

comptime {
    if (@import("builtin").target.cpu.arch.isSpirV()) {
        @export(&faceBcs, .{ .name = "face_bcs" });
    }
    // Vulkan guarantees only 128 bytes of push-constant range, and this is the
    // largest of the kernels' -- it carries a whole freestream state. Sizing the
    // arrays for three dimensions rather than the case's costs 16 bytes of it.
    if (@sizeOf(PushConstants) > 128) {
        @compileError("face_bcs push constants exceed the guaranteed 128-byte range");
    }
}

/// Which condition a boundary flux point carries. The host maps
/// `cfg.BoundaryCondition` onto these; anything it cannot map sends the whole
/// step to the CPU, so the kernel never sees a code it does not know.
pub const Code = enum(u32) {
    /// Impose the freestream
    sup_in = 0,
    /// Everything leaves: extrapolate
    sup_out = 1,
    /// Riemann-invariant far field
    characteristic = 2,
    /// Reflect the normal momentum
    slip_wall = 3,
};

pub const PushConstants = extern struct {
    n_gfpts: u32,
    n_gfpts_int: u32,
    n_gfpts_bnd: u32,
    n_vars: u32,
    gamma: f64,
    rho_fs: f64,
    p_fs: f64,
    /// Sized for the largest dimension so one layout serves both builds; a 2D
    /// kernel reads the first two, and the first four.
    vel_fs: [3]f64,
    u_fs: [5]f64,
};

pub const WgSize = extern struct {
    pub const x: u32 = 128;
    pub const y: u32 = 1;
    pub const z: u32 = 1;
};

const F64Array = @SpirvType(.{ .runtime_array = f64 });
const U32Array = @SpirvType(.{ .runtime_array = u32 });
const F64Buf = extern struct { data: F64Array };
const U32Buf = extern struct { data: U32Array };

const u = @extern(*addrspace(.storage_buffer) F64Buf, .{
    .name = "u",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 0 } },
});
const norm = @extern(*addrspace(.storage_buffer) const F64Buf, .{
    .name = "norm",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 1 } },
});
const bc_code = @extern(*addrspace(.storage_buffer) const U32Buf, .{
    .name = "bc_code",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 2 } },
});
const u_ldg = @extern(*addrspace(.storage_buffer) F64Buf, .{
    .name = "u_ldg",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 3 } },
});

const pc = @extern(*addrspace(.push_constant) const PushConstants, .{ .name = "pc" });

/// The dimension this build is for, injected by `build.zig`.
///
/// Comptime rather than a push constant so every loop below unrolls and the
/// state stays in registers. The host imports this file for its push-constant
/// layout alone and has no `kernel_dims` module, so on that side reaching for
/// `n_dims` is a compile error rather than a quietly wrong 2.
const n_dims: usize = if (@import("builtin").target.cpu.arch.isSpirV())
    @import("kernel_dims").n_dims
else
    @compileError("n_dims is device-only; the host half of this file is dimension-independent");

/// Euler carries density, momentum per dimension, and total energy.
const n_vars = n_dims + 2;

/// Total energy's index in the conserved state, one past the last momentum.
const i_energy = n_dims + 1;

/// Riemann-invariant far field, after PyFR. One incoming and one outgoing
/// invariant are combined into a state that lets waves leave without
/// reflecting, and matches the freestream on inflow.
fn characteristic(ul: [n_vars]f64, nrm: [n_dims]f64) [n_vars]f64 {
    const gam = pc.gamma;
    const gm1 = gam - 1.0;

    const rho_l = ul[0];
    var vn_l: f64 = 0.0;
    var vn_r: f64 = 0.0;
    for (0..n_dims) |d| {
        vn_l += ul[1 + d] / rho_l * nrm[d];
        vn_r += pc.vel_fs[d] * nrm[d];
    }

    var mom_sq: f64 = 0.0;
    for (0..n_dims) |d| mom_sq += ul[1 + d] * ul[1 + d];
    const press_l = gm1 * (ul[i_energy] - 0.5 * mom_sq / rho_l);
    const press_r = pc.p_fs;

    const c_l = @sqrt(gam * press_l / rho_l);
    const c_r = @sqrt(gam * press_r / pc.rho_fs);

    // Outgoing invariant, unless the far field is supersonic into the domain
    const supersonic = @abs(vn_r) >= c_r;
    const r_l = if (supersonic and vn_l >= 0.0)
        vn_r + 2.0 / gm1 * c_r
    else
        vn_l + 2.0 / gm1 * c_l;

    const r_b = if (supersonic and vn_l < 0.0)
        vn_l - 2.0 / gm1 * c_l
    else
        vn_r - 2.0 / gm1 * c_r;

    const c_star = 0.25 * gm1 * (r_l - r_b);
    const vn_star = 0.5 * (r_l + r_b);

    // Entropy comes from whichever side the flow arrives from
    var rho_r = c_star * c_star / gam;
    var vel: [n_dims]f64 = undefined;
    if (vn_l < 0.0) { // inflow
        rho_r *= pow(pc.rho_fs, gam) / press_r;
        for (0..n_dims) |d| vel[d] = pc.vel_fs[d] + (vn_star - vn_r) * nrm[d];
    } else { // outflow
        rho_r *= pow(rho_l, gam) / press_l;
        for (0..n_dims) |d| vel[d] = ul[1 + d] / rho_l + (vn_star - vn_l) * nrm[d];
    }
    rho_r = pow(rho_r, 1.0 / gm1);

    var ur: [n_vars]f64 = undefined;
    ur[0] = rho_r;
    var ke: f64 = 0.0;
    for (0..n_dims) |d| {
        ur[1 + d] = rho_r * vel[d];
        ke += vel[d] * vel[d];
    }
    const press_b = rho_r / gam * c_star * c_star;
    ur[i_energy] = press_b / gm1 + 0.5 * rho_r * ke;
    return ur;
}

fn faceBcs() callconv(.{ .spirv_kernel = .{ .x = WgSize.x, .y = WgSize.y, .z = WgSize.z } }) void {
    const i = std.spirv.global_invocation_id[0];
    if (i >= pc.n_gfpts_bnd) return;

    const gf = pc.n_gfpts_int + i;

    var ul: [n_vars]f64 = undefined;
    for (0..n_vars) |n| ul[n] = u.data[gf + pc.n_gfpts * n];

    var nrm: [n_dims]f64 = undefined;
    for (0..n_dims) |d| nrm[d] = norm.data[gf + pc.n_gfpts * d];

    var ur: [n_vars]f64 = undefined;
    const code = bc_code.data[i];

    if (code == @backingInt(Code.sup_in)) {
        for (0..n_vars) |n| ur[n] = pc.u_fs[n];
    } else if (code == @backingInt(Code.sup_out)) {
        ur = ul;
    } else if (code == @backingInt(Code.characteristic)) {
        ur = characteristic(ul, nrm);
    } else {
        // slip_wall / symmetry: reflect the normal momentum, so the Riemann
        // average carries none through the wall. Density and energy are
        // unchanged because the momentum magnitude is.
        var mom_n: f64 = 0.0;
        for (0..n_dims) |d| mom_n += ul[1 + d] * nrm[d];
        ur[0] = ul[0];
        for (0..n_dims) |d| ur[1 + d] = ul[1 + d] - 2.0 * mom_n * nrm[d];
        ur[i_energy] = ul[i_energy];
    }

    for (0..n_vars) |n| {
        const dst = gf + pc.n_gfpts * (n + pc.n_vars);
        u.data[dst] = ur[n];
        u_ldg.data[dst] = ur[n];
    }
}

const std = @import("std");
const pow = @import("mathf64.zig").pow;

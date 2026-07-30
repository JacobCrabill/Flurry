//! Tests for the FR solver core: geometry transforms, operator applications,
//! physical fluxes and the RK update.
//!
//! The centrepiece is "uniform flow has zero residual": a constant state on a
//! Cartesian mesh must produce an exactly zero flux divergence once the DFR
//! interior and correction terms are combined. That exercises the whole
//! residual chain apart from the face plumbing, which is still stubbed.

const std = @import("std");
const testing = std.testing;

const cfg = @import("config.zig");
const flux = @import("flux.zig");
const geo_mod = @import("geo.zig");
const Geo = geo_mod.Geo;
const FaceType = geo_mod.FaceType;
const solver_mod = @import("solver.zig");
const Solver = solver_mod.Solver;
const RkScheme = solver_mod.RkScheme;
const Faces = @import("faces.zig").Faces;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// A config for a Cartesian mesh over [0, xmax] x [0, ymax].
fn testConfig(
    order: u8,
    equation: cfg.Equation,
    viscous: bool,
    nx: u32,
    ny: u32,
) cfg.Config {
    var config: cfg.Config = undefined;
    config.core = .{ .n_dims = 2, .mesh_file = "", .order = order };
    config.equation = .{
        .equation = equation,
        .viscous = viscous,
        .advdiff_A = .{ 1.0, 0.5, 0.0 },
        .advdiff_D = 0.1,
    };
    config.time = .{ .dt_scheme = .rk44, .n_steps = 1, .dt = 1e-3 };
    config.test_case = .{ .test_case = 0, .err_field = 0, .n_qpts_1d = 0 };
    config.flux = .{};
    config.gas_properties = .{};
    config.freestream = .{ .mach_fs = 0.3, .norm_fs = .{ 1.0, 0.0, 0.0 }, .fix_vis = true };
    config.wall_conditions = .{};
    config.boundary_conditions = .{};
    config.create_mesh = .{
        .nx = nx,
        .ny = ny,
        .xmin = 0.0,
        .xmax = 2.0,
        .ymin = 0.0,
        .ymax = 3.0,
        // CreateMeshConfig defaults every side to periodic; make that opt-in so
        // a test only exercises the periodic path when it means to.
        .bc_bottom = .characteristic,
        .bc_right = .characteristic,
        .bc_top = .characteristic,
        .bc_left = .characteristic,
        .bc_front = .characteristic,
        .bc_back = .characteristic,
    };
    return config;
}

/// Build a Cartesian mesh with full connectivity and its global flux points.
fn testMesh(gpa: std.mem.Allocator, config: *const cfg.Config) !Geo {
    var mesh: Geo = .{ .gpa = gpa, .io = undefined, .config = config.* };
    errdefer mesh.deinit();
    try mesh.createMesh();
    try mesh.processConnectivity();
    try mesh.setupGlobalFpts(@as(usize, config.core.order) + 1);
    return mesh;
}

/// Same, but with the vertices moved so the mapping is *non-affine*: cells
/// become general quadrilaterals whose Jacobian varies within the cell.
///
/// This is the case that distinguishes a correct metric adjugate from its
/// transpose -- on an affine mesh the adjugate is constant and a transposed one
/// still preserves free-stream.
fn testMeshDistorted(gpa: std.mem.Allocator, config: *const cfg.Config) !Geo {
    var mesh: Geo = .{ .gpa = gpa, .io = undefined, .config = config.* };
    errdefer mesh.deinit();
    try mesh.createMesh();

    // Both displacements depend on both coordinates, so cells do not stay
    // parallelograms: a separable perturbation would leave every cell affine.
    for (0..mesh.n_verts) |iv| {
        const x = mesh.xv.get(iv, 0);
        const y = mesh.xv.get(iv, 1);
        mesh.xv.at(iv, 0).* = x + 0.10 * @sin(1.3 * x) * @sin(0.9 * y);
        mesh.xv.at(iv, 1).* = y + 0.10 * @sin(1.1 * x) * @sin(0.7 * y);
    }

    try mesh.processConnectivity();
    try mesh.setupGlobalFpts(@as(usize, config.core.order) + 1);
    return mesh;
}

// ---------------------------------------------------------------------------
// Flux functions
// ---------------------------------------------------------------------------

test "nVars per equation set" {
    try testing.expectEqual(@as(usize, 1), flux.nVars(.adv_diff, 2));
    try testing.expectEqual(@as(usize, 1), flux.nVars(.adv_diff, 3));
    try testing.expectEqual(@as(usize, 4), flux.nVars(.euler_ns, 2));
    try testing.expectEqual(@as(usize, 5), flux.nVars(.euler_ns, 3));
}

test "advection flux is A times u" {
    const p: flux.FlowParams = .{ .adv_vel = .{ 1.5, -0.5, 0.0 }, .diff_coeff = 0.25 };

    const f = flux.convAdvDiff(2, .{2.0}, p);
    try testing.expectApproxEqAbs(@as(f64, 3.0), f[0][0], 1e-15);
    try testing.expectApproxEqAbs(@as(f64, -1.0), f[0][1], 1e-15);

    // Diffusion subtracts D * grad(u)
    var fv: [1][2]f64 = .{f[0]};
    flux.viscAdvDiffAdd(2, .{.{ 4.0, 8.0 }}, &fv, p);
    try testing.expectApproxEqAbs(@as(f64, 3.0 - 1.0), fv[0][0], 1e-15);
    try testing.expectApproxEqAbs(@as(f64, -1.0 - 2.0), fv[0][1], 1e-15);
}

test "Euler flux matches its closed form" {
    const p: flux.FlowParams = .{ .gamma = 1.4 };

    const rho: f64 = 1.2;
    const u: f64 = 30.0;
    const v: f64 = -10.0;
    const press: f64 = 101325.0;
    const e = press / (p.gamma - 1.0) + 0.5 * rho * (u * u + v * v);

    const state: [4]f64 = .{ rho, rho * u, rho * v, e };
    const res = flux.convEulerNS(2, state, p);

    try testing.expectApproxEqRel(press, res.p, 1e-12);
    try testing.expectApproxEqRel(press, flux.pressure(2, state, p.gamma), 1e-12);

    const h = (e + press) / rho;
    // Mass
    try testing.expectApproxEqRel(rho * u, res.f[0][0], 1e-12);
    try testing.expectApproxEqRel(rho * v, res.f[0][1], 1e-12);
    // x-momentum
    try testing.expectApproxEqRel(rho * u * u + press, res.f[1][0], 1e-12);
    try testing.expectApproxEqRel(rho * u * v, res.f[1][1], 1e-12);
    // y-momentum
    try testing.expectApproxEqRel(rho * u * v, res.f[2][0], 1e-12);
    try testing.expectApproxEqRel(rho * v * v + press, res.f[2][1], 1e-12);
    // Energy
    try testing.expectApproxEqRel(rho * u * h, res.f[3][0], 1e-12);
    try testing.expectApproxEqRel(rho * v * h, res.f[3][1], 1e-12);
}

test "viscous NS flux vanishes for a uniform state and is symmetric" {
    const p: flux.FlowParams = .{ .gamma = 1.4, .prandtl = 0.72, .mu = 1e-3, .fix_vis = true };
    const state: [4]f64 = .{ 1.0, 0.3, -0.2, 2.5 };

    // No gradient, no viscous flux
    {
        var f: [4][2]f64 = @splat(@splat(0.0));
        flux.viscEulerNSAdd(2, state, @splat(@splat(0.0)), &f, p);
        for (f) |row| for (row) |v| try testing.expectApproxEqAbs(@as(f64, 0.0), v, 1e-15);
    }

    // Pure rotation (du/dy = -dv/dx) produces no stress
    {
        // Build conserved gradients that give du/dy = 1, dv/dx = -1 with all
        // other primitive gradients zero. With drho = 0, d(rho u)/dy = rho du/dy.
        var du: [4][2]f64 = @splat(@splat(0.0));
        du[1][1] = 1.0; // d(rho u)/dy
        du[2][0] = -1.0; // d(rho v)/dx

        var f: [4][2]f64 = @splat(@splat(0.0));
        flux.viscEulerNSAdd(2, state, du, &f, p);
        // tau_xy = mu (du/dy + dv/dx) = 0
        try testing.expectApproxEqAbs(@as(f64, 0.0), f[1][1], 1e-14);
        try testing.expectApproxEqAbs(@as(f64, 0.0), f[2][0], 1e-14);
    }

    // Pure shear du/dy = 1 gives tau_xy = mu, subtracted from the flux
    {
        var du: [4][2]f64 = @splat(@splat(0.0));
        du[1][1] = 1.0;

        var f: [4][2]f64 = @splat(@splat(0.0));
        flux.viscEulerNSAdd(2, state, du, &f, p);
        try testing.expectApproxEqRel(-p.mu, f[1][1], 1e-12);
        try testing.expectApproxEqRel(-p.mu, f[2][0], 1e-12);
    }
}

test "wave speed is |Vn| + a for Euler and |An| for advection" {
    {
        const p: flux.FlowParams = .{ .adv_vel = .{ 3.0, 4.0, 0.0 } };
        const s = flux.waveSpeed(2, .adv_diff, .{ 1.0, 0, 0, 0 }, .{ 0.6, 0.8 }, p);
        try testing.expectApproxEqRel(@as(f64, 5.0), s, 1e-14);
    }
    {
        const p: flux.FlowParams = .{ .gamma = 1.4 };
        const rho: f64 = 1.0;
        const press: f64 = 1.0;
        const vx: f64 = 0.5;
        const e = press / (p.gamma - 1.0) + 0.5 * rho * vx * vx;
        const a = @sqrt(p.gamma * press / rho);
        const s = flux.waveSpeed(2, .euler_ns, .{ rho, rho * vx, 0.0, e }, .{ 1.0, 0.0 }, p);
        try testing.expectApproxEqRel(vx + a, s, 1e-12);
    }
}

test "FlowParams nondimensionalization" {
    var config = testConfig(2, .euler_ns, true, 2, 2);
    config.freestream.Re_fs = 100.0;
    config.freestream.fix_vis = true;
    const p = flux.FlowParams.fromConfig(&config);

    // The nondimensional freestream is a unit-speed flow along norm_fs, and mu
    // reduces to 1/Re.
    try testing.expectApproxEqRel(@as(f64, 1.0), p.rho_fs, 1e-12);
    try testing.expectApproxEqRel(@as(f64, 1.0), p.vel_fs[0], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), p.vel_fs[1], 1e-15);
    try testing.expectApproxEqRel(1.0 / config.freestream.Re_fs, p.mu, 1e-12);

    // The freestream Mach number must come back out of the state
    const state = p.freestreamState(2, .euler_ns);
    const press = flux.pressure(2, state, p.gamma);
    const a = @sqrt(p.gamma * press / state[0]);
    const speed = state[1] / state[0];
    try testing.expectApproxEqRel(config.freestream.mach_fs, speed / a, 1e-10);
}

// ---------------------------------------------------------------------------
// RK schemes
// ---------------------------------------------------------------------------

test "RK tableaux are consistent" {
    for ([_]cfg.DtScheme{ .euler, .rk44, .rkJ }) |scheme| {
        const rk = try RkScheme.fromConfig(scheme);
        try testing.expectEqual(rk.n_stages, rk.c.len);

        if (rk.combines_stages) {
            // The weights of a consistent explicit RK method sum to one
            var sum: f64 = 0.0;
            for (rk.beta) |b| sum += b;
            try testing.expectApproxEqAbs(@as(f64, 1.0), sum, 1e-14);
            try testing.expectEqual(rk.n_stages, rk.beta.len);
            try testing.expectEqual(rk.n_stages - 1, rk.alpha.len);
        } else {
            try testing.expectEqual(rk.n_stages, rk.alpha.len);
        }
        // The first stage is always evaluated at the start of the step
        try testing.expectApproxEqAbs(@as(f64, 0.0), rk.c[0], 1e-14);
    }

    // Unported schemes must say so rather than silently misbehave
    for ([_]cfg.DtScheme{ .rk54, .steady }) |scheme| {
        try testing.expectError(error.NotImplemented, RkScheme.fromConfig(scheme));
    }
}

// ---------------------------------------------------------------------------
// Geometry transforms
// ---------------------------------------------------------------------------

test "transforms on a uniform Cartesian mesh" {
    const gpa = testing.allocator;

    const nx: u32 = 4;
    const ny: u32 = 3;
    var config = testConfig(3, .euler_ns, false, nx, ny);
    var mesh = try testMesh(gpa, &config);
    defer mesh.deinit();

    var s = try Solver.init(gpa, &config, &mesh, .{});
    defer s.deinit();

    const ele = &s.quad.ele;
    try testing.expectEqual(@as(usize, nx * ny), s.n_eles);
    try testing.expectEqual(@as(usize, 4), s.n_vars);

    // Cell size, and the constant Jacobian of an affine map
    const dx = (config.create_mesh.?.xmax - config.create_mesh.?.xmin) / @as(f64, nx);
    const dy = (config.create_mesh.?.ymax - config.create_mesh.?.ymin) / @as(f64, ny);
    const det = (dx / 2.0) * (dy / 2.0);

    for (0..s.n_eles) |e| {
        for (0..ele.n_spts) |spt| {
            try testing.expectApproxEqRel(det, s.jaco_det_spts.get(spt, e), 1e-12);
            // jaco = diag(dx/2, dy/2)
            try testing.expectApproxEqRel(dx / 2.0, s.jaco_spts.get(0, spt, 0, e), 1e-12);
            try testing.expectApproxEqRel(dy / 2.0, s.jaco_spts.get(1, spt, 1, e), 1e-12);
            try testing.expectApproxEqAbs(@as(f64, 0.0), s.jaco_spts.get(0, spt, 1, e), 1e-13);
            try testing.expectApproxEqAbs(@as(f64, 0.0), s.jaco_spts.get(1, spt, 0, e), 1e-13);
        }
        try testing.expectApproxEqRel(dx * dy, s.vol[e], 1e-12);
    }

    // Total volume is the domain area
    var total: f64 = 0.0;
    for (s.vol) |v| total += v;
    try testing.expectApproxEqRel(
        (config.create_mesh.?.xmax - config.create_mesh.?.xmin) *
            (config.create_mesh.?.ymax - config.create_mesh.?.ymin),
        total,
        1e-12,
    );

    // Solution point coordinates land inside their own cell
    for (0..s.n_eles) |e| {
        for (0..ele.n_spts) |spt| {
            const x = s.coord_spts.get(spt, 0, e);
            const y = s.coord_spts.get(spt, 1, e);
            try testing.expect(x > config.create_mesh.?.xmin and x < config.create_mesh.?.xmax);
            try testing.expect(y > config.create_mesh.?.ymin and y < config.create_mesh.?.ymax);
        }
    }
}

test "face normals and areas" {
    const gpa = testing.allocator;

    const nx: u32 = 2;
    const ny: u32 = 2;
    var config = testConfig(2, .euler_ns, false, nx, ny);
    var mesh = try testMesh(gpa, &config);
    defer mesh.deinit();

    var s = try Solver.init(gpa, &config, &mesh, .{});
    defer s.deinit();

    const ele = &s.quad.ele;
    const dx = 2.0 / @as(f64, nx);
    const dy = 3.0 / @as(f64, ny);

    for (0..s.n_eles) |e| {
        for (0..ele.n_fpts) |fpt| {
            const face = fpt / ele.n_fpts_per_face;
            // On a Cartesian mesh the physical normals equal the reference ones
            const want: [2]f64 = switch (face) {
                0 => .{ 0, -1 },
                1 => .{ 1, 0 },
                2 => .{ 0, 1 },
                else => .{ -1, 0 },
            };
            try testing.expectApproxEqAbs(want[0], s.norm_fpts.get(fpt, 0, e), 1e-13);
            try testing.expectApproxEqAbs(want[1], s.norm_fpts.get(fpt, 1, e), 1e-13);

            // dA is the physical/reference face measure ratio: a face of
            // physical length L maps from a reference face of length 2.
            const want_da = if (face == 0 or face == 2) dx / 2.0 else dy / 2.0;
            try testing.expectApproxEqRel(want_da, s.d_a_fpts.get(fpt, e), 1e-12);
        }

        // Closed-surface identity: the weighted sum of outward area vectors
        // over a closed cell is zero.
        var sum: [2]f64 = .{ 0, 0 };
        for (0..ele.n_fpts) |fpt| {
            const w = ele.weights_fpts[fpt % ele.n_fpts_per_face];
            const da = s.d_a_fpts.get(fpt, e);
            for (0..2) |d| sum[d] += w * da * s.norm_fpts.get(fpt, d, e);
        }
        try testing.expectApproxEqAbs(@as(f64, 0.0), sum[0], 1e-12);
        try testing.expectApproxEqAbs(@as(f64, 0.0), sum[1], 1e-12);
    }
}

test "solver rejects meshes it has no element for" {
    const gpa = testing.allocator;

    // 3D needs hexes, which are not implemented
    {
        var config = testConfig(2, .euler_ns, false, 2, 2);
        config.core.n_dims = 3;
        var mesh = try testMesh(gpa, &config);
        defer mesh.deinit();
        try testing.expectError(error.UnsupportedDimension, Solver.init(gpa, &config, &mesh, .{}));
    }

    // Triangles have no element type yet either
    {
        var config = testConfig(2, .euler_ns, false, 2, 2);
        var mesh = try testMesh(gpa, &config);
        defer mesh.deinit();
        mesh.ctype.items[0] = .tri;
        try testing.expectError(error.UnsupportedCellType, Solver.init(gpa, &config, &mesh, .{}));
    }
}

// ---------------------------------------------------------------------------
// Residual assembly
// ---------------------------------------------------------------------------

/// Fill `f_comm` the way a completed face gather would for a *single-valued*
/// flux: extrapolate the physical flux to each flux point and dot it with the
/// outward area vector.
///
/// For a smooth flux this is exactly what any consistent Riemann solver returns,
/// so it lets the residual chain be checked without the face plumbing.
fn fillExactFComm(s: *Solver, comptime nd: usize) void {
    const ele = &s.quad.ele;
    const n_vars = s.n_vars;

    for (0..ele.n_fpts) |fpt| {
        for (0..s.n_eles) |e| {
            var u: [nd + 2]f64 = @splat(0.0);
            for (0..n_vars) |n| u[n] = s.u_fpts.get(fpt, n, e);

            var f: [nd + 2][nd]f64 = @splat(@splat(0.0));
            switch (s.config.equation.equation) {
                .adv_diff => {
                    const fc = flux.convAdvDiff(nd, .{u[0]}, s.params);
                    f[0] = fc[0];
                },
                .euler_ns => {
                    const fc = flux.convEulerNS(nd, u, s.params);
                    f = fc.f;
                },
            }

            const da = s.d_a_fpts.get(fpt, e);
            for (0..n_vars) |n| {
                var fn_: f64 = 0.0;
                for (0..nd) |d| fn_ += f[n][d] * s.norm_fpts.get(fpt, d, e);
                s.f_comm.at(fpt, n, e).* = fn_ * da;
            }
        }
    }
}

test "uniform flow has exactly zero residual" {
    const gpa = testing.allocator;

    // Both equation sets, several orders, and a non-square cell aspect ratio
    for ([_]cfg.Equation{ .adv_diff, .euler_ns }) |equation| {
        for ([_]u8{ 1, 2, 3, 4 }) |order| {
            var config = testConfig(order, equation, false, 3, 2);
            var mesh = try testMesh(gpa, &config);
            defer mesh.deinit();

            var s = try Solver.init(gpa, &config, &mesh, .{});
            defer s.deinit();

            try s.initializeU();
            try s.extrapolateU();
            try s.computeFluxSpts();
            try s.computeDivFSpts(0);
            fillExactFComm(&s, 2);
            try s.computeDivFFpts(0);

            // Uniform state, uniform flux, zero divergence. The interior term
            // alone is *not* zero -- it takes the DFR correction to cancel it.
            const ele = &s.quad.ele;
            for (0..ele.n_spts) |spt| {
                for (0..s.n_vars) |n| {
                    for (0..s.n_eles) |e| {
                        try testing.expectApproxEqAbs(
                            @as(f64, 0.0),
                            s.divf_spts.get(0, spt, n, e),
                            1e-9,
                        );
                    }
                }
            }
        }
    }
}

test "linear flow reproduces its analytic divergence" {
    const gpa = testing.allocator;

    // A linear advected field: u = 1 + x + 2y, so div(A u) = A . grad(u)
    var config = testConfig(3, .adv_diff, false, 3, 2);
    var mesh = try testMesh(gpa, &config);
    defer mesh.deinit();

    var s = try Solver.init(gpa, &config, &mesh, .{});
    defer s.deinit();

    const ele = &s.quad.ele;
    for (0..ele.n_spts) |spt| {
        for (0..s.n_eles) |e| {
            const x = s.coord_spts.get(spt, 0, e);
            const y = s.coord_spts.get(spt, 1, e);
            s.u_spts.at(spt, 0, e).* = 1.0 + x + 2.0 * y;
        }
    }

    try s.extrapolateU();
    try s.computeFluxSpts();
    try s.computeDivFSpts(0);
    fillExactFComm(&s, 2);
    try s.computeDivFFpts(0);

    // divF is the *reference*-space divergence, so it carries a factor of |J|
    const a = s.params.adv_vel;
    const exact = a[0] * 1.0 + a[1] * 2.0;

    for (0..ele.n_spts) |spt| {
        for (0..s.n_eles) |e| {
            const got = s.divf_spts.get(0, spt, 0, e) / s.jaco_det_spts.get(spt, e);
            try testing.expectApproxEqAbs(exact, got, 1e-9);
        }
    }
}

test "extrapolateU matches applying oppE by hand" {
    const gpa = testing.allocator;

    var config = testConfig(2, .euler_ns, false, 2, 2);
    var mesh = try testMesh(gpa, &config);
    defer mesh.deinit();

    var s = try Solver.init(gpa, &config, &mesh, .{});
    defer s.deinit();

    const ele = &s.quad.ele;

    // Distinct values everywhere, so a transposed index would show up
    var seed: u64 = 12345;
    for (0..ele.n_spts) |spt| {
        for (0..s.n_vars) |n| {
            for (0..s.n_eles) |e| {
                seed = seed *% 6364136223846793005 +% 1;
                s.u_spts.at(spt, n, e).* = @floatFromInt((seed >> 40) % 1000);
            }
        }
    }

    try s.extrapolateU();

    for (0..ele.n_fpts) |fpt| {
        for (0..s.n_vars) |n| {
            for (0..s.n_eles) |e| {
                var sum: f64 = 0.0;
                for (0..ele.n_spts) |spt| {
                    sum += ele.oppE.get(fpt, spt) * s.u_spts.get(spt, n, e);
                }
                try testing.expectApproxEqRel(sum, s.u_fpts.get(fpt, n, e), 1e-12);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Time stepping
// ---------------------------------------------------------------------------

/// Stand in for a residual of exactly `c` in physical space. `divf_spts` holds
/// the reference-space divergence, so it carries a factor of |J|.
fn setConstantResidual(s: *Solver, stage: usize, c: f64) void {
    const ele = &s.quad.ele;
    for (0..ele.n_spts) |spt| {
        for (0..s.n_vars) |n| {
            for (0..s.n_eles) |e| {
                s.divf_spts.at(stage, spt, n, e).* = c * s.jaco_det_spts.get(spt, e);
            }
        }
    }
}

test "RK stages integrate du/dt = -c exactly" {
    const gpa = testing.allocator;

    // With a residual held constant, every consistent RK method must advance
    // the solution by exactly -dt * c over one step.
    for ([_]cfg.DtScheme{ .euler, .rk44, .rkJ }) |scheme| {
        var config = testConfig(2, .adv_diff, false, 2, 2);
        config.time.dt_scheme = scheme;
        var mesh = try testMesh(gpa, &config);
        defer mesh.deinit();

        var s = try Solver.init(gpa, &config, &mesh, .{});
        defer s.deinit();

        const ele = &s.quad.ele;
        const c: f64 = 3.0;
        @memset(s.u_spts.data, 5.0);

        // Drive the update directly, with divF standing in for a constant
        // residual, since computeResidual still needs the face plumbing.
        const prev = s.flow_time;
        if (s.rk.n_stages > 1) @memcpy(s.u_ini.data, s.u_spts.data);

        const n_steps = if (s.rk.combines_stages) s.rk.n_stages - 1 else s.rk.n_stages;
        for (0..n_steps) |stage| {
            setConstantResidual(&s, stage, c);
            s.rkStage(stage);
        }
        if (s.rk.combines_stages) {
            setConstantResidual(&s, s.rk.n_stages - 1, c);
            s.rkCombine();
        }
        s.flow_time = prev + s.dt;

        for (0..ele.n_spts) |spt| {
            for (0..s.n_eles) |e| {
                try testing.expectApproxEqRel(
                    5.0 - s.dt * c,
                    s.u_spts.get(spt, 0, e),
                    1e-12,
                );
            }
        }
        try testing.expectApproxEqRel(config.time.dt.?, s.flow_time, 1e-15);
    }
}

test "global flux points cover every element face exactly once" {
    const gpa = testing.allocator;

    const order: u8 = 2;
    var config = testConfig(order, .euler_ns, false, 3, 2);
    var mesh = try testMesh(gpa, &config);
    defer mesh.deinit();

    const nfpf: usize = @as(usize, order) + 1;
    try testing.expectEqual(nfpf, mesh.n_fpts_per_face);
    try testing.expectEqual(mesh.n_int_faces * nfpf, mesh.n_gfpts_int);
    try testing.expectEqual(mesh.n_bnd_faces * nfpf, mesh.n_gfpts_bnd);
    try testing.expectEqual(mesh.n_gfpts_int + mesh.n_gfpts_bnd, mesh.n_gfpts);

    // Count how many element flux points land on each gfpt, per slot
    const seen = try gpa.alloc([2]usize, mesh.n_gfpts);
    defer gpa.free(seen);
    @memset(seen, .{ 0, 0 });

    for (0..mesh.n_eles) |e| {
        for (0..mesh.n_fpts_per_ele) |fpt| {
            const gf = mesh.fpt2gfpt.get(fpt, e);
            try testing.expect(gf != geo_mod.none); // no collapsed quads here
            try testing.expect(gf < mesh.n_gfpts);
            seen[gf][mesh.fpt2gfpt_slot.get(fpt, e)] += 1;
        }
    }

    // An interior gfpt is written by exactly one flux point on each side; a
    // boundary gfpt by exactly one, on the left.
    for (0..mesh.n_gfpts_int) |gf| {
        try testing.expectEqual(@as(usize, 1), seen[gf][0]);
        try testing.expectEqual(@as(usize, 1), seen[gf][1]);
    }
    for (mesh.n_gfpts_int..mesh.n_gfpts) |gf| {
        try testing.expectEqual(@as(usize, 1), seen[gf][0]);
        try testing.expectEqual(@as(usize, 0), seen[gf][1]);
    }

    // Every boundary gfpt knows its boundary, and every gfpt its face
    for (mesh.n_gfpts_int..mesh.n_gfpts) |gf| {
        const bnd = mesh.gfpt2bnd.items[gf - mesh.n_gfpts_int];
        try testing.expect(bnd < mesh.n_bounds or bnd == geo_mod.none);
        try testing.expectEqual(FaceType.boundary, mesh.face_type.items[mesh.gfpt2face.items[gf]]);
    }
    for (0..mesh.n_gfpts_int) |gf| {
        try testing.expectEqual(FaceType.internal, mesh.face_type.items[mesh.gfpt2face.items[gf]]);
    }
}

test "the two sides of every interface meet at the same point" {
    const gpa = testing.allocator;

    // This is the assumption `setupGlobalFpts` makes: two cells sharing a face
    // traverse it in opposite directions, so the right side's flux points are
    // the left side's reversed. Checked geometrically, at several orders and on
    // a mesh with a non-unit cell aspect ratio.
    for ([_]u8{ 1, 2, 3, 4 }) |order| {
        var config = testConfig(order, .euler_ns, false, 4, 3);
        var mesh = try testMesh(gpa, &config);
        defer mesh.deinit();

        var s = try Solver.init(gpa, &config, &mesh, .{});
        defer s.deinit();

        try testing.expect(s.fptPairingError() < 1e-13);
    }
}

test "faces receive their geometry from the elements" {
    const gpa = testing.allocator;

    var config = testConfig(2, .euler_ns, false, 3, 3);
    var mesh = try testMesh(gpa, &config);
    defer mesh.deinit();

    var s = try Solver.init(gpa, &config, &mesh, .{});
    defer s.deinit();

    // Every gfpt got a unit normal and a positive face scaling on both sides
    for (0..mesh.n_gfpts) |gf| {
        var mag: f64 = 0.0;
        for (0..s.n_dims) |d| {
            const c = s.faces.norm.get(d, gf);
            mag += c * c;
        }
        try testing.expectApproxEqAbs(@as(f64, 1.0), @sqrt(mag), 1e-13);
        try testing.expect(s.faces.d_a.get(0, gf) > 0.0);
        if (gf < mesh.n_gfpts_int) try testing.expect(s.faces.d_a.get(1, gf) > 0.0);
    }

    // On a Cartesian mesh both sides of an interface agree on the face measure
    for (0..mesh.n_gfpts_int) |gf| {
        try testing.expectApproxEqRel(s.faces.d_a.get(0, gf), s.faces.d_a.get(1, gf), 1e-12);
    }
}

test "free-stream is preserved on cells away from the boundary" {
    const gpa = testing.allocator;

    // The end-to-end check: a uniform state, pushed through the full residual
    // chain including the interface coupling, must give exactly zero divergence.
    //
    // Boundary conditions are still stubbed, so the right-hand state at a
    // boundary flux point is meaningless and the cells touching one are
    // excluded. Cells with four interior faces are unaffected.
    for ([_]cfg.Equation{ .adv_diff, .euler_ns }) |equation| {
        for ([_]u8{ 1, 2, 3 }) |order| {
            var config = testConfig(order, equation, false, 4, 4);
            var mesh = try testMesh(gpa, &config);
            defer mesh.deinit();

            var s = try Solver.init(gpa, &config, &mesh, .{});
            defer s.deinit();

            try s.initializeU();

            // computeResidual itself still stops at applyBcs, so drive the
            // chain directly.
            try s.extrapolateU();
            s.scatterUToFaces();
            try s.computeFluxSpts();
            try s.computeDivFSpts(0);
            s.faces.computeCommonF();
            s.gatherCommonFFromFaces();
            try s.computeDivFFpts(0);

            const ele = &s.quad.ele;
            var n_interior: usize = 0;
            for (0..s.n_eles) |e| {
                var on_bnd = false;
                for (0..mesh.c2nf.items[e]) |j| {
                    if (mesh.c2b.get(e, j) != 0) on_bnd = true;
                }
                if (on_bnd) continue;
                n_interior += 1;

                for (0..ele.n_spts) |spt| {
                    for (0..s.n_vars) |n| {
                        try testing.expectApproxEqAbs(
                            @as(f64, 0.0),
                            s.divf_spts.get(0, spt, n, e),
                            1e-9,
                        );
                    }
                }
            }
            // A 4x4 mesh has a 2x2 block of fully interior cells
            try testing.expectEqual(@as(usize, 4), n_interior);
        }
    }
}

/// A doubly-periodic Cartesian mesh, ready for a solver.
fn testMeshPeriodic(gpa: std.mem.Allocator, config: *cfg.Config) !Geo {
    config.create_mesh.?.bc_bottom = .periodic;
    config.create_mesh.?.bc_top = .periodic;
    config.create_mesh.?.bc_left = .periodic;
    config.create_mesh.?.bc_right = .periodic;
    return testMesh(gpa, config);
}

test "a periodic mesh has no boundary faces at all" {
    const gpa = testing.allocator;

    // Every face of a doubly-periodic box is interior: the boundary faces pair
    // up across the domain and merge.
    var config = testConfig(2, .euler_ns, false, 4, 3);
    var mesh = try testMeshPeriodic(gpa, &config);
    defer mesh.deinit();

    // A 4x3 torus: 4*3 faces normal to x, 4*3 normal to y
    try testing.expectEqual(@as(usize, 24), mesh.n_faces);
    try testing.expectEqual(@as(usize, 24), mesh.n_int_faces);
    try testing.expectEqual(@as(usize, 0), mesh.n_bnd_faces);
    try testing.expectEqual(@as(usize, 0), mesh.n_gfpts_bnd);

    // Every cell has four neighbours, and no cell face is marked as boundary
    for (0..mesh.n_eles) |e| {
        for (0..mesh.c2nf.items[e]) |j| {
            try testing.expectEqual(@as(usize, 0), mesh.c2b.get(e, j));
            try testing.expect(mesh.c2c.get(e, j) != geo_mod.none);
        }
    }

    // Every face is shared by exactly two cells
    for (0..mesh.n_faces) |ff| {
        try testing.expect(mesh.f2c.get(ff, 0) != geo_mod.none);
        try testing.expect(mesh.f2c.get(ff, 1) != geo_mod.none);
        try testing.expectEqual(FaceType.internal, mesh.face_type.items[ff]);
    }
}

test "periodic pairing wraps opposite sides of the domain" {
    const gpa = testing.allocator;

    var config = testConfig(3, .euler_ns, false, 4, 3);
    var mesh = try testMeshPeriodic(gpa, &config);
    defer mesh.deinit();

    var s = try Solver.init(gpa, &config, &mesh, .{});
    defer s.deinit();

    const lx = config.create_mesh.?.xmax - config.create_mesh.?.xmin;
    const ly = config.create_mesh.?.ymax - config.create_mesh.?.ymin;

    // The two sides of a periodic interface are a full domain length apart, so
    // the plain coordinate check must see that -- and the periodic-aware one
    // must not.
    try testing.expect(s.fptPairingError() > 0.5 * @min(lx, ly));
    try testing.expect(s.fptPairingErrorPeriodic() < 1e-12);

    // Cells on opposite edges really are neighbours: cell 0 is at the
    // lower-left, and its -x and -y neighbours are on the far side.
    const nx: usize = 4;
    const ny: usize = 3;
    var found_wrap = false;
    for (0..mesh.c2nf.items[0]) |j| {
        const nb = mesh.c2c.get(0, j);
        // createMesh numbers cells x-outer, y-inner
        if (nb == (nx - 1) * ny or nb == ny - 1) found_wrap = true;
    }
    try testing.expect(found_wrap);
}

test "free-stream is preserved on a periodic mesh" {
    const gpa = testing.allocator;

    // With no boundaries at all, a uniform state must be preserved exactly --
    // and unlike the characteristic far field, nothing here can quietly damp an
    // error back towards the freestream.
    for ([_]cfg.Equation{ .adv_diff, .euler_ns }) |equation| {
        for ([_]u8{ 1, 2, 3 }) |order| {
            var config = testConfig(order, equation, false, 4, 3);
            var mesh = try testMeshPeriodic(gpa, &config);
            defer mesh.deinit();

            var s = try Solver.init(gpa, &config, &mesh, .{});
            defer s.deinit();

            try s.initializeU();
            try s.computeResidual(0);

            const ele = &s.quad.ele;
            for (0..s.n_eles) |e| {
                for (0..ele.n_spts) |spt| {
                    for (0..s.n_vars) |n| {
                        try testing.expectApproxEqAbs(
                            @as(f64, 0.0),
                            s.divf_spts.get(0, spt, n, e),
                            1e-9,
                        );
                    }
                }
            }
        }
    }
}

test "a periodic mesh conserves mass exactly" {
    const gpa = testing.allocator;

    // A closed domain: the total of every conserved variable cannot change,
    // because the flux out of one cell is the flux into its neighbour. This is
    // the strongest statement about the periodic wiring -- it fails if any
    // interface is paired to the wrong partner or with the wrong sign.
    var config = testConfig(3, .euler_ns, false, 4, 3);
    var mesh = try testMeshPeriodic(gpa, &config);
    defer mesh.deinit();

    var s = try Solver.init(gpa, &config, &mesh, .{});
    defer s.deinit();

    // A non-uniform state, so the fluxes are genuinely doing something
    const ele = &s.quad.ele;
    for (0..ele.n_spts) |spt| {
        for (0..s.n_eles) |e| {
            const x = s.coord_spts.get(spt, 0, e);
            const y = s.coord_spts.get(spt, 1, e);
            const bump = 0.1 * @sin(std.math.pi * x) * @cos(2.0 * std.math.pi * y / 3.0);
            s.u_spts.at(spt, 0, e).* = 1.0 + bump;
            s.u_spts.at(spt, 1, e).* = 0.3 * (1.0 + bump);
            s.u_spts.at(spt, 2, e).* = -0.1 * (1.0 + bump);
            s.u_spts.at(spt, 3, e).* = 2.5 + bump;
        }
    }

    try s.computeResidual(0);

    // The integral of the divergence over a closed domain is zero for every
    // variable. divF already carries |J|, so the reference weights suffice.
    for (0..s.n_vars) |n| {
        var total: f64 = 0.0;
        var scale: f64 = 0.0;
        for (0..ele.n_spts) |spt| {
            for (0..s.n_eles) |e| {
                const r = ele.weights_spts[spt] * s.divf_spts.get(0, spt, n, e);
                total += r;
                scale += @abs(r);
            }
        }
        try testing.expect(scale > 1e-6); // the fluxes are non-trivial
        try testing.expectApproxEqAbs(@as(f64, 0.0), total / scale, 1e-12);
    }
}

test "a periodic direction spanning two cells is rejected" {
    const gpa = testing.allocator;

    // Identifying periodic faces by vertex set is exact only while no two
    // distinct faces share one. Two cells across a periodic direction breaks
    // that: the boundary line running along that direction closes into a
    // two-edge loop, and both edges span the same vertex pair. Note this shows
    // up in the faces normal to the *other* direction, so it does not matter
    // whether that one is periodic as well.
    for ([_][2]u32{ .{ 2, 4 }, .{ 4, 2 }, .{ 2, 2 } }) |n| {
        var config = testConfig(2, .euler_ns, false, n[0], n[1]);
        config.create_mesh.?.bc_left = .periodic;
        config.create_mesh.?.bc_right = .periodic;
        if (n[1] == 2) {
            config.create_mesh.?.bc_bottom = .periodic;
            config.create_mesh.?.bc_top = .periodic;
        }

        var mesh: Geo = .{ .gpa = gpa, .io = undefined, .config = config };
        defer mesh.deinit();
        try mesh.createMesh();

        try testing.expectError(error.PeriodicDirectionTooThin, mesh.processConnectivity());
    }

    // Three cells across is enough
    {
        var config = testConfig(2, .euler_ns, false, 3, 3);
        var mesh = try testMeshPeriodic(gpa, &config);
        defer mesh.deinit();
        try testing.expectEqual(@as(usize, 0), mesh.n_bnd_faces);
    }
}

test "a direction is only periodic if its boundary faces say so" {
    const gpa = testing.allocator;

    // A channel: periodic in x, walled in y, and deliberately only two cells
    // thick. Matching vertices purely by separation would call y periodic too,
    // because the two ends of the domain are exactly the bounding-box y extent
    // apart and both lie on the x-periodic boundary. That misreading fuses the
    // channel's two x-boundary edges and rejects a perfectly good mesh.
    const nx: usize = 4;
    const ny: usize = 2;

    var config = testConfig(3, .euler_ns, false, nx, ny);
    config.create_mesh.?.bc_left = .periodic;
    config.create_mesh.?.bc_right = .periodic;

    var mesh = try testMesh(gpa, &config);
    defer mesh.deinit();

    // Only the walls are left: x wrapped, y did not.
    try testing.expectEqual(nx * ny * 4, 2 * mesh.n_int_faces + mesh.n_bnd_faces);
    try testing.expectEqual(2 * nx, mesh.n_bnd_faces);

    // Every cell has a neighbour in x, and the end cells wrap to each other
    const ymin = config.create_mesh.?.ymin;
    const ymax = config.create_mesh.?.ymax;
    for (0..mesh.n_faces) |ff| {
        if (mesh.face_type.items[ff] == .internal) continue;
        // A boundary face must be one of the y walls
        var y: f64 = 0.0;
        for (0..mesh.f2nv.items[ff]) |j| y += mesh.xv.get(mesh.f2v.get(ff, j), 1);
        y /= @floatFromInt(mesh.f2nv.items[ff]);
        try testing.expect(@abs(y - ymin) < 1e-12 or @abs(y - ymax) < 1e-12);
    }

    // And the solver still preserves free-stream across the wrap
    var s = try Solver.init(gpa, &config, &mesh, .{});
    defer s.deinit();

    try testing.expect(s.fptPairingErrorPeriodic() < 1e-12);

    try s.initializeU();
    try s.computeResidual(0);

    const ele = &s.quad.ele;
    for (0..s.n_eles) |e| {
        for (0..ele.n_spts) |spt| {
            for (0..s.n_vars) |n| {
                try testing.expectApproxEqAbs(@as(f64, 0.0), s.divf_spts.get(0, spt, n, e), 1e-9);
            }
        }
    }
}

test "free-stream is preserved over the whole domain" {
    const gpa = testing.allocator;

    // The definitive check on the boundary conditions: with a characteristic
    // far-field on every side, a uniform freestream must give exactly zero
    // residual *everywhere*, boundary cells included. Any inconsistency between
    // the interior scheme and a boundary state shows up here.
    for ([_]u8{ 1, 2, 3 }) |order| {
        for ([_]bool{ false, true }) |distorted| {
            var config = testConfig(order, .euler_ns, false, 4, 4);
            config.create_mesh.?.xmax = 4.0;
            config.create_mesh.?.ymax = 4.0;
            config.create_mesh.?.bc_bottom = .characteristic;
            config.create_mesh.?.bc_top = .characteristic;
            config.create_mesh.?.bc_left = .characteristic;
            config.create_mesh.?.bc_right = .characteristic;

            var mesh = if (distorted)
                try testMeshDistorted(gpa, &config)
            else
                try testMesh(gpa, &config);
            defer mesh.deinit();

            var s = try Solver.init(gpa, &config, &mesh, .{});
            defer s.deinit();

            try s.initializeU();
            try s.computeResidual(0);

            const ele = &s.quad.ele;
            for (0..s.n_eles) |e| {
                for (0..ele.n_spts) |spt| {
                    for (0..s.n_vars) |n| {
                        try testing.expectApproxEqAbs(
                            @as(f64, 0.0),
                            s.divf_spts.get(0, spt, n, e),
                            1e-9,
                        );
                    }
                }
            }

            // And a full step leaves it alone
            const before = try gpa.dupe(f64, s.u_spts.data);
            defer gpa.free(before);
            try s.update();
            for (before, s.u_spts.data) |a, b| {
                try testing.expectApproxEqAbs(a, b, 1e-11);
            }
        }
    }
}

test "a slip wall reflects without generating mass" {
    const gpa = testing.allocator;

    // Slip walls top and bottom, characteristic in and out: a uniform flow
    // parallel to the walls must also be preserved exactly.
    var config = testConfig(3, .euler_ns, false, 4, 4);
    config.create_mesh.?.xmax = 4.0;
    config.create_mesh.?.ymax = 4.0;
    config.create_mesh.?.bc_bottom = .slip_wall;
    config.create_mesh.?.bc_top = .slip_wall;
    config.create_mesh.?.bc_left = .characteristic;
    config.create_mesh.?.bc_right = .characteristic;
    config.freestream.norm_fs = .{ 1.0, 0.0, 0.0 }; // along the walls

    var mesh = try testMesh(gpa, &config);
    defer mesh.deinit();

    var s = try Solver.init(gpa, &config, &mesh, .{});
    defer s.deinit();

    try s.initializeU();
    try s.computeResidual(0);

    const ele = &s.quad.ele;
    for (0..s.n_eles) |e| {
        for (0..ele.n_spts) |spt| {
            for (0..s.n_vars) |n| {
                try testing.expectApproxEqAbs(@as(f64, 0.0), s.divf_spts.get(0, spt, n, e), 1e-9);
            }
        }
    }
}

test "solver requires the global flux point layout" {
    const gpa = testing.allocator;

    var config = testConfig(2, .euler_ns, false, 2, 2);

    // Connectivity processed, but setupGlobalFpts never run
    {
        var mesh: Geo = .{ .gpa = gpa, .io = undefined, .config = config };
        defer mesh.deinit();
        try mesh.createMesh();
        try mesh.processConnectivity();
        try testing.expectError(error.ConnectivityNotProcessed, Solver.init(gpa, &config, &mesh, .{}));
    }

    // Laid out for the wrong order
    {
        var mesh: Geo = .{ .gpa = gpa, .io = undefined, .config = config };
        defer mesh.deinit();
        try mesh.createMesh();
        try mesh.processConnectivity();
        try mesh.setupGlobalFpts(7);
        try testing.expectError(error.ConnectivityNotProcessed, Solver.init(gpa, &config, &mesh, .{}));
    }

    // And it cannot be built before the connectivity exists
    {
        var mesh: Geo = .{ .gpa = gpa, .io = undefined, .config = config };
        defer mesh.deinit();
        try mesh.createMesh();
        try testing.expectError(error.ConnectivityNotProcessed, mesh.setupGlobalFpts(3));
    }
}

test "computeResidual and update run end to end" {
    const gpa = testing.allocator;

    // Characteristic far-field on every side: the whole residual chain now runs,
    // boundary conditions included.
    var config = testConfig(2, .euler_ns, false, 3, 3);
    config.create_mesh.?.bc_bottom = .characteristic;
    config.create_mesh.?.bc_top = .characteristic;
    config.create_mesh.?.bc_left = .characteristic;
    config.create_mesh.?.bc_right = .characteristic;

    var mesh = try testMesh(gpa, &config);
    defer mesh.deinit();

    var s = try Solver.init(gpa, &config, &mesh, .{});
    defer s.deinit();
    try s.initializeU();

    try testing.expect(s.faces.n_gfpts_bnd > 0);
    try s.computeResidual(0);
    try s.update();

    try testing.expectEqual(@as(u32, 1), s.current_iter);
    try testing.expectApproxEqRel(config.time.dt.?, s.flow_time, 1e-14);
}

test "residualNorm is zero for a zero residual and scales linearly" {
    const gpa = testing.allocator;

    var config = testConfig(2, .adv_diff, false, 2, 2);
    var mesh = try testMesh(gpa, &config);
    defer mesh.deinit();

    var s = try Solver.init(gpa, &config, &mesh, .{});
    defer s.deinit();

    var norms: [1]f64 = undefined;

    s.residualNorm(0, &norms);
    try testing.expectApproxEqAbs(@as(f64, 0.0), norms[0], 1e-15);

    // divF carries a factor of |J|, so a residual of |J| * c has norm c
    const ele = &s.quad.ele;
    const c: f64 = 2.5;
    for (0..ele.n_spts) |spt| {
        for (0..s.n_eles) |e| {
            s.divf_spts.at(0, spt, 0, e).* = c * s.jaco_det_spts.get(spt, e);
        }
    }
    s.residualNorm(0, &norms);
    try testing.expectApproxEqRel(c, norms[0], 1e-12);
}

// ---------------------------------------------------------------------------
// Faces
// ---------------------------------------------------------------------------

/// A bare `Faces` with `n` interior flux points, unit normals along +x and
/// unit face scaling on both sides.
fn testFaces(gpa: std.mem.Allocator, config: *const cfg.Config, n: usize) !Faces {
    const params = flux.FlowParams.fromConfig(config);
    var f = try Faces.init(gpa, config, params, n, 0);
    for (0..n) |gf| {
        f.norm.at(0, gf).* = 1.0;
        f.norm.at(1, gf).* = 0.0;
        f.d_a.at(0, gf).* = 1.0;
        f.d_a.at(1, gf).* = 1.0;
    }
    return f;
}

test "Rusanov reduces to the exact flux for a continuous state" {
    const gpa = testing.allocator;
    const config = testConfig(2, .euler_ns, false, 2, 2);

    var f = try testFaces(gpa, &config, 3);
    defer f.deinit();

    // Identical states: the dissipation term drops out and the common flux is
    // just F . n.
    const state: [4]f64 = .{ 1.0, 0.4, -0.2, 3.0 };
    for (0..f.n_gfpts) |gf| {
        for (0..f.n_vars) |n| {
            f.u.at(0, n, gf).* = state[n];
            f.u.at(1, n, gf).* = state[n];
        }
    }

    f.computeCommonF();

    const exact = flux.convEulerNS(2, state, f.params);
    for (0..f.n_gfpts) |gf| {
        for (0..f.n_vars) |n| {
            try testing.expectApproxEqRel(exact.f[n][0], f.f_comm.get(0, n, gf), 1e-12);
        }
    }
}

test "Rusanov is conservative and upwinds" {
    const gpa = testing.allocator;
    const config = testConfig(2, .adv_diff, false, 2, 2);

    var f = try testFaces(gpa, &config, 2);
    defer f.deinit();

    f.u.at(0, 0, 0).* = 1.0;
    f.u.at(1, 0, 0).* = 3.0;
    f.u.at(0, 0, 1).* = 3.0;
    f.u.at(1, 0, 1).* = 1.0;

    f.computeCommonF();

    // The two sides must see equal and opposite flux, or mass is created at the
    // interface. This is the sign convention `oppDiv_fpts` relies on: each
    // element's own outward normal is already folded into the operator.
    for (0..f.n_gfpts) |gf| {
        for (0..f.n_vars) |n| {
            try testing.expectApproxEqAbs(
                f.f_comm.get(0, n, gf),
                -f.f_comm.get(1, n, gf),
                1e-14,
            );
        }
    }

    // Advection is along +x at speed 1 with the normal along +x, so the exact
    // upwind flux is the left state.
    try testing.expectApproxEqRel(@as(f64, 1.0), f.f_comm.get(0, 0, 0), 1e-12);
    try testing.expectApproxEqRel(@as(f64, 3.0), f.f_comm.get(0, 0, 1), 1e-12);

    // The wave speed is |A . n|
    for (f.wave_sp) |w| try testing.expectApproxEqRel(@as(f64, 1.0), w, 1e-12);
}

test "Rusanov dissipation is scaled by rus_k" {
    const gpa = testing.allocator;

    // rus_k = 1 removes the dissipation entirely, leaving a central flux
    var config = testConfig(2, .adv_diff, false, 2, 2);
    config.flux.rus_k = 1.0;

    var f = try testFaces(gpa, &config, 1);
    defer f.deinit();

    f.u.at(0, 0, 0).* = 1.0;
    f.u.at(1, 0, 0).* = 3.0;
    f.computeCommonF();

    // Central: 0.5 * (A ul + A ur) = 0.5 * (1 + 3) = 2
    try testing.expectApproxEqRel(@as(f64, 2.0), f.f_comm.get(0, 0, 0), 1e-12);
}

test "common solution is single-valued and biased by ldg_b" {
    const gpa = testing.allocator;

    var config = testConfig(2, .adv_diff, true, 2, 2);
    config.flux.ldg_b = 0.5; // fully biased to the left state

    var f = try testFaces(gpa, &config, 1);
    defer f.deinit();

    // computeCommonU reads u_ldg, which scatterUToFaces fills alongside u
    f.u_ldg.at(0, 0, 0).* = 2.0;
    f.u_ldg.at(1, 0, 0).* = 8.0;
    f.computeCommonU();

    try testing.expectApproxEqRel(@as(f64, 2.0), f.u_comm.get(0, 0, 0), 1e-12);
    // Both sides see the same value, signed for each one's outward normal
    try testing.expectApproxEqRel(@as(f64, -2.0), f.u_comm.get(1, 0, 0), 1e-12);

    // Centred biasing averages the two
    config.flux.ldg_b = 0.0;
    f.config = &config;
    f.computeCommonU();
    try testing.expectApproxEqRel(@as(f64, 5.0), f.u_comm.get(0, 0, 0), 1e-12);
}

/// A `Faces` whose every flux point is a boundary point on boundary 0, with the
/// given condition and an outward normal along +x.
fn testBndFaces(
    gpa: std.mem.Allocator,
    config: *const cfg.Config,
    bc: *const [1]cfg.BoundaryCondition,
    gfpt2bnd: []const usize,
) !Faces {
    const params = flux.FlowParams.fromConfig(config);
    var f = try Faces.init(gpa, config, params, gfpt2bnd.len, gfpt2bnd.len);
    f.gfpt2bnd = gfpt2bnd;
    f.bc_list = bc;
    for (0..f.n_gfpts) |gf| {
        f.norm.at(0, gf).* = 1.0;
        f.norm.at(1, gf).* = 0.0;
        f.d_a.at(0, gf).* = 1.0;
    }
    return f;
}

/// Write `state` into the interior (left) side of every flux point.
fn setLeftState(f: *Faces, state: []const f64) void {
    for (0..f.n_gfpts) |gf| {
        for (state, 0..) |v, n| {
            f.u.at(0, n, gf).* = v;
            if (f.config.equation.viscous) f.u_ldg.at(0, n, gf).* = v;
        }
    }
}

test "sup_out extrapolates, sup_in imposes the freestream" {
    const gpa = testing.allocator;
    const config = testConfig(2, .euler_ns, false, 2, 2);
    const params = flux.FlowParams.fromConfig(&config);
    const u_fs = params.freestreamState(2, .euler_ns);
    const interior = [4]f64{ 1.3, 0.4, -0.2, 3.1 };

    {
        var f = try testBndFaces(gpa, &config, &.{.sup_out}, &.{0});
        defer f.deinit();
        setLeftState(&f, &interior);
        try f.applyBcs();
        for (0..4) |n| try testing.expectApproxEqRel(interior[n], f.u.get(1, n, 0), 1e-14);
    }

    {
        var f = try testBndFaces(gpa, &config, &.{.sup_in}, &.{0});
        defer f.deinit();
        setLeftState(&f, &interior);
        try f.applyBcs();
        for (0..4) |n| try testing.expectApproxEqAbs(u_fs[n], f.u.get(1, n, 0), 1e-14);
    }
}

test "slip wall is impermeable" {
    const gpa = testing.allocator;
    const config = testConfig(2, .euler_ns, false, 2, 2);

    for ([_]cfg.BoundaryCondition{ .slip_wall, .symmetry }) |bc| {
        var f = try testBndFaces(gpa, &config, &.{bc}, &.{0});
        defer f.deinit();

        const interior = [4]f64{ 1.3, 0.4, -0.2, 3.1 };
        setLeftState(&f, &interior);
        try f.applyBcs();

        // Normal momentum reflected, tangential kept, density and energy alike
        try testing.expectApproxEqRel(interior[0], f.u.get(1, 0, 0), 1e-14);
        try testing.expectApproxEqRel(-interior[1], f.u.get(1, 1, 0), 1e-14);
        try testing.expectApproxEqRel(interior[2], f.u.get(1, 2, 0), 1e-14);
        try testing.expectApproxEqRel(interior[3], f.u.get(1, 3, 0), 1e-14);

        // The physical requirement: no mass crosses the wall. Pressure still
        // acts on it, so the normal momentum flux must not vanish.
        f.computeCommonF();
        try testing.expectApproxEqAbs(@as(f64, 0.0), f.f_comm.get(0, 0, 0), 1e-14);
        try testing.expect(@abs(f.f_comm.get(0, 1, 0)) > 1e-3);
    }
}

test "characteristic far-field preserves the freestream exactly" {
    const gpa = testing.allocator;
    const config = testConfig(2, .euler_ns, false, 2, 2);
    const params = flux.FlowParams.fromConfig(&config);
    const u_fs = params.freestreamState(2, .euler_ns);

    // The property free-stream preservation depends on: if the interior state
    // already is the freestream, the Riemann invariants must return it unchanged
    // whichever way the boundary faces.
    for ([_][2]f64{ .{ 1, 0 }, .{ -1, 0 }, .{ 0, 1 }, .{ 0.6, -0.8 } }) |n| {
        var f = try testBndFaces(gpa, &config, &.{.characteristic}, &.{0});
        defer f.deinit();
        f.norm.at(0, 0).* = n[0];
        f.norm.at(1, 0).* = n[1];

        setLeftState(&f, &u_fs);
        try f.applyBcs();

        for (0..4) |v| try testing.expectApproxEqAbs(u_fs[v], f.u.get(1, v, 0), 1e-12);
    }
}

test "characteristic far-field lets an outgoing perturbation leave" {
    const gpa = testing.allocator;
    const config = testConfig(2, .euler_ns, false, 2, 2);
    const params = flux.FlowParams.fromConfig(&config);
    var u_fs = params.freestreamState(2, .euler_ns);

    // Outflow (normal along the flow): the boundary state should track the
    // interior, not snap back to the freestream.
    var f = try testBndFaces(gpa, &config, &.{.characteristic}, &.{0});
    defer f.deinit();

    u_fs[0] *= 1.05; // denser than freestream
    setLeftState(&f, &u_fs);
    try f.applyBcs();

    const rho_r = f.u.get(1, 0, 0);
    const rho_ref = params.freestreamState(2, .euler_ns)[0];
    try testing.expect(rho_r > rho_ref);
}

test "no-slip walls prescribe the wall state and reflect for the Riemann solve" {
    const gpa = testing.allocator;
    var config = testConfig(2, .euler_ns, true, 2, 2);
    config.wall_conditions.mach_wall = 0.0; // stationary wall
    const params = flux.FlowParams.fromConfig(&config);

    const rho: f64 = 1.2;
    const interior = [4]f64{ rho, rho * 0.3, rho * -0.1, 3.0 };

    {
        var f = try testBndFaces(gpa, &config, &.{.isothermal_noslip}, &.{0});
        defer f.deinit();
        setLeftState(&f, &interior);
        try f.applyBcs();

        // Ghost state: velocity reversed so the average is zero
        try testing.expectApproxEqRel(rho, f.u.get(1, 0, 0), 1e-14);
        try testing.expectApproxEqRel(-interior[1], f.u.get(1, 1, 0), 1e-13);
        try testing.expectApproxEqRel(-interior[2], f.u.get(1, 2, 0), 1e-13);

        // Prescribed state: at rest, at the wall temperature
        try testing.expectApproxEqAbs(@as(f64, 0.0), f.u_ldg.get(1, 1, 0), 1e-14);
        try testing.expectApproxEqAbs(@as(f64, 0.0), f.u_ldg.get(1, 2, 0), 1e-14);
        const cv_t = params.r_ref / (params.gamma - 1.0) * params.t_wall;
        try testing.expectApproxEqRel(rho * cv_t, f.u_ldg.get(1, 3, 0), 1e-12);
    }

    {
        var f = try testBndFaces(gpa, &config, &.{.adiabatic_noslip}, &.{0});
        defer f.deinit();
        setLeftState(&f, &interior);
        try f.applyBcs();

        try testing.expectApproxEqRel(-interior[1], f.u.get(1, 1, 0), 1e-13);
        try testing.expectApproxEqAbs(@as(f64, 0.0), f.u_ldg.get(1, 1, 0), 1e-14);

        // Energy is extrapolated, not prescribed: the ghost state keeps the
        // interior's kinetic energy, the prescribed one has none.
        try testing.expectApproxEqRel(interior[3], f.u.get(1, 3, 0), 1e-13);
        const ke = 0.5 * (interior[1] * interior[1] + interior[2] * interior[2]) / rho;
        try testing.expectApproxEqRel(interior[3] - ke, f.u_ldg.get(1, 3, 0), 1e-12);
    }
}

test "adiabatic wall removes the normal temperature gradient" {
    const gpa = testing.allocator;
    const config = testConfig(2, .euler_ns, true, 2, 2);

    var f = try testBndFaces(gpa, &config, &.{.adiabatic_noslip}, &.{0});
    defer f.deinit();

    const rho: f64 = 1.0;
    setLeftState(&f, &.{ rho, 0.0, 0.0, 2.5 });

    // A purely wall-normal energy gradient, no density or momentum gradient.
    // With the velocity zero this is entirely a temperature gradient, so the
    // boundary must cancel it.
    f.du.at(0, 0, 3, 0).* = 1.0;
    try f.applyBcsGrad();

    try testing.expectApproxEqAbs(@as(f64, 0.0), f.du.get(1, 0, 3, 0), 1e-13);
    // Density gradient is extrapolated untouched
    try testing.expectApproxEqAbs(@as(f64, 0.0), f.du.get(1, 0, 0, 0), 1e-14);

    // A purely *tangential* energy gradient carries no heat through the wall,
    // so it must survive.
    @memset(f.du.data, 0.0);
    f.du.at(0, 1, 3, 0).* = 1.0;
    try f.applyBcsGrad();
    try testing.expectApproxEqRel(@as(f64, 1.0), f.du.get(1, 1, 3, 0), 1e-13);
}

test "unsupported and unmatched boundaries are reported" {
    const gpa = testing.allocator;

    // Periodic faces need processPeriodicBoundaries, which is not ported
    {
        const config = testConfig(2, .euler_ns, false, 2, 2);
        var f = try testBndFaces(gpa, &config, &.{.periodic}, &.{0});
        defer f.deinit();
        setLeftState(&f, &.{ 1.0, 0.1, 0.0, 2.0 });
        try testing.expectError(error.UnsupportedBoundaryCondition, f.applyBcs());
    }

    // A boundary face that matched no declared boundary
    {
        const config = testConfig(2, .euler_ns, false, 2, 2);
        var f = try testBndFaces(gpa, &config, &.{.sup_out}, &.{geo_mod.none});
        defer f.deinit();
        setLeftState(&f, &.{ 1.0, 0.1, 0.0, 2.0 });
        try testing.expectError(error.UnmatchedBoundaryFace, f.applyBcs());
    }

    // A no-slip wall needs a viscous run, a slip wall an inviscid one
    {
        const config = testConfig(2, .euler_ns, false, 2, 2);
        var f = try testBndFaces(gpa, &config, &.{.adiabatic_noslip}, &.{0});
        defer f.deinit();
        setLeftState(&f, &.{ 1.0, 0.1, 0.0, 2.0 });
        try testing.expectError(error.WallConditionMismatch, f.applyBcs());
    }
    {
        const config = testConfig(2, .euler_ns, true, 2, 2);
        var f = try testBndFaces(gpa, &config, &.{.slip_wall}, &.{0});
        defer f.deinit();
        setLeftState(&f, &.{ 1.0, 0.1, 0.0, 2.0 });
        try testing.expectError(error.WallConditionMismatch, f.applyBcs());
    }
}

test "the metric adjugate satisfies adj . jaco = |J| I" {
    const gpa = testing.allocator;

    // The defining identity of the adjugate, which also pins the index
    // convention: `jaco(dr, pt, dp, e)` is d(x_dp)/d(xi_dr), so the Jacobian
    // matrix M[j][k] = d(x_j)/d(xi_k) is `jaco.get(k, pt, j, e)`, and
    //     sum_j adj[i][j] * M[j][k] == |J| delta_ik
    //
    // Getting the two off-diagonal entries of adj the wrong way round satisfies
    // this only when the mapping is affine, which is why this runs on a
    // distorted mesh.
    var config = testConfig(3, .euler_ns, false, 4, 4);
    config.create_mesh.?.xmax = 4.0;
    config.create_mesh.?.ymax = 4.0;

    var mesh = try testMeshDistorted(gpa, &config);
    defer mesh.deinit();

    var s = try Solver.init(gpa, &config, &mesh, .{});
    defer s.deinit();

    const ele = &s.quad.ele;
    for (0..s.n_eles) |e| {
        for (0..ele.n_spts) |spt| {
            const det = s.jaco_det_spts.get(spt, e);
            for (0..s.n_dims) |i| {
                for (0..s.n_dims) |k| {
                    var sum: f64 = 0.0;
                    for (0..s.n_dims) |j| {
                        sum += s.inv_jaco_spts.get(i, spt, j, e) * s.jaco_spts.get(k, spt, j, e);
                    }
                    const want: f64 = if (i == k) det else 0.0;
                    try testing.expectApproxEqAbs(want, sum, 1e-12 * @max(1.0, det));
                }
            }
        }
    }
}

test "free-stream is preserved on a non-affine mesh" {
    const gpa = testing.allocator;

    // The regression test for the metric adjugate. A transposed adjugate passes
    // every affine-mesh check and fails here by O(1), growing with order.
    for ([_]cfg.Equation{ .adv_diff, .euler_ns }) |equation| {
        for ([_]u8{ 1, 2, 3 }) |order| {
            var config = testConfig(order, equation, false, 4, 4);
            config.create_mesh.?.xmax = 4.0;
            config.create_mesh.?.ymax = 4.0;

            var mesh = try testMeshDistorted(gpa, &config);
            defer mesh.deinit();

            var s = try Solver.init(gpa, &config, &mesh, .{});
            defer s.deinit();

            // Without these two properties the test proves nothing:
            //
            //  - |J| must vary *within* a cell, or the mapping is affine and the
            //    metric identity holds trivially;
            //  - the adjugate must be asymmetric, or transposing it is a no-op.
            var worst_spread: f64 = 0.0;
            var worst_asym: f64 = 0.0;
            for (0..s.n_eles) |e| {
                var lo = std.math.inf(f64);
                var hi = -std.math.inf(f64);
                for (0..s.quad.ele.n_spts) |spt| {
                    lo = @min(lo, s.jaco_det_spts.get(spt, e));
                    hi = @max(hi, s.jaco_det_spts.get(spt, e));
                    worst_asym = @max(worst_asym, @abs(
                        s.inv_jaco_spts.get(0, spt, 1, e) - s.inv_jaco_spts.get(1, spt, 0, e),
                    ));
                }
                worst_spread = @max(worst_spread, hi - lo);
            }
            try testing.expect(worst_spread > 1e-3);
            try testing.expect(worst_asym > 1e-2);

            try s.initializeU();
            try s.extrapolateU();
            s.scatterUToFaces();
            try s.computeFluxSpts();
            try s.computeDivFSpts(0);
            s.faces.computeCommonF();
            s.gatherCommonFFromFaces();
            try s.computeDivFFpts(0);

            const ele = &s.quad.ele;
            for (0..s.n_eles) |e| {
                var on_bnd = false;
                for (0..mesh.c2nf.items[e]) |j| {
                    if (mesh.c2b.get(e, j) != 0) on_bnd = true;
                }
                if (on_bnd) continue;

                for (0..ele.n_spts) |spt| {
                    for (0..s.n_vars) |n| {
                        try testing.expectApproxEqAbs(
                            @as(f64, 0.0),
                            s.divf_spts.get(0, spt, n, e),
                            1e-9,
                        );
                    }
                }
            }
        }
    }
}

test "interfaces pair up on a non-affine mesh" {
    const gpa = testing.allocator;

    var config = testConfig(3, .euler_ns, false, 4, 4);
    config.create_mesh.?.xmax = 4.0;
    config.create_mesh.?.ymax = 4.0;

    var mesh = try testMeshDistorted(gpa, &config);
    defer mesh.deinit();

    var s = try Solver.init(gpa, &config, &mesh, .{});
    defer s.deinit();

    try testing.expect(s.fptPairingError() < 1e-12);

    // Both sides of an interface still agree on the physical face measure, even
    // though each cell's mapping is different
    for (0..mesh.n_gfpts_int) |gf| {
        try testing.expectApproxEqRel(s.faces.d_a.get(0, gf), s.faces.d_a.get(1, gf), 1e-11);
    }
}

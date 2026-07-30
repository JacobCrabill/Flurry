//! Interface (flux point) coupling: common solution, common normal flux, and
//! boundary conditions.
//!
//! Every interface appears once here, as a *global flux point* (gfpt) with a
//! left (slot 0) and right (slot 1) side. Interior gfpts come first; boundary
//! ones occupy the tail, where the right-hand state is manufactured by
//! `applyBcs` rather than gathered from a neighbour. Elements scatter into and
//! gather from these through `geo.fpt2gfpt` / `fpt2gfpt_slot`.
//!
//! Two states are carried at a boundary. `u` is the *ghost* state the Rusanov
//! flux sees, chosen so the Riemann average gives the right physics -- a no-slip
//! wall reflects the velocity so the average is zero. `u_ldg` is the state
//! *prescribed* to the viscous flux, which for that same wall is the wall
//! velocity itself. They coincide on interior faces and for most conditions.
//!
//! ## Not ported
//!
//! - `.periodic`: a periodic face is really an interior face, and pairing the
//!   two sides needs Flurry-cpp's `processPeriodicBoundaries`. Reported as
//!   `error.UnsupportedBoundaryCondition` rather than silently mishandled.
//! - MPI processor boundaries.
//! - The `rus_bias` / `LDG_bias` machinery: ZEFR uses the "prescribed" Rusanov
//!   variant only for its implicit method, so the explicit path always takes the
//!   ghost-state branch implemented here.
//!
//! Ported from ZEFR's `faces.cpp`.

pub const Error = error{
    OutOfMemory,
    /// A boundary condition with no implementation: periodic faces need
    /// `processPeriodicBoundaries`, which is not ported
    UnsupportedBoundaryCondition,
    /// A boundary face that matched no declared boundary, so there is no
    /// condition to apply
    UnmatchedBoundaryFace,
    /// A no-slip wall on an inviscid run, or a slip wall on a viscous one
    WallConditionMismatch,
};

pub const Faces = struct {
    gpa: std.mem.Allocator,
    config: *const Config,
    params: flux.FlowParams,

    n_dims: usize = 0,
    n_vars: usize = 0,

    /// Global flux points: interior interfaces plus boundary faces
    n_gfpts: usize = 0,
    /// Interior gfpts, which come first
    n_gfpts_int: usize = 0,
    /// Of those, the ones on a mesh boundary (they occupy the tail)
    n_gfpts_bnd: usize = 0,

    // ---- Interface state, (slot, var, gfpt) ----

    /// Solution on each side of the interface
    u: Array3(f64) = .empty,

    /// Common (single-valued) solution, for the viscous gradient correction
    u_comm: Array3(f64) = .empty,

    /// The state the *viscous* (LDG) flux sees. Equal to `u` everywhere except
    /// at boundaries, where a wall's prescribed state differs from the ghost
    /// state the convective flux needs -- a no-slip wall prescribes zero
    /// velocity but reflects it for the Riemann solve.
    u_ldg: Array3(f64) = .empty,

    /// Common normal flux, already scaled by `dA` and signed per slot so that
    /// each element can add it directly through `oppDiv_fpts`
    f_comm: Array3(f64) = .empty,

    /// Solution gradient on each side, (slot, dim, var, gfpt)
    du: Array4(f64) = .empty,

    // ---- Interface geometry ----

    /// Outward unit normal of the left element, (dim, gfpt)
    norm: Matrix(f64) = .empty,

    /// Physical/reference face scaling on each side, (slot, gfpt).
    /// Slot 1 differs from slot 0 whenever the two elements disagree on the
    /// reference-space size of the shared face.
    d_a: Matrix(f64) = .empty,

    /// Physical coordinates of each flux point, (dim, gfpt)
    coord: Matrix(f64) = .empty,

    /// Maximum wave speed at each flux point, for the time-step limit
    wave_sp: []f64 = &.{},

    // ---- Boundary information, borrowed from the mesh ----

    /// Boundary index of each boundary gfpt, indexed by `gfpt - n_gfpts_int`
    gfpt2bnd: []const usize = &.{},

    /// Condition applied on each boundary, indexed by boundary
    bc_list: []const cfg.BoundaryCondition = &.{},

    pub fn init(
        gpa: std.mem.Allocator,
        config: *const Config,
        params: flux.FlowParams,
        n_gfpts: usize,
        n_gfpts_bnd: usize,
    ) Error!Faces {
        const n_dims: usize = config.core.n_dims;
        const n_vars = flux.nVars(config.equation.equation, n_dims);

        var f: Faces = .{
            .gpa = gpa,
            .config = config,
            .params = params,
            .n_dims = n_dims,
            .n_vars = n_vars,
            .n_gfpts = n_gfpts,
            .n_gfpts_int = n_gfpts - n_gfpts_bnd,
            .n_gfpts_bnd = n_gfpts_bnd,
        };
        errdefer f.deinit();

        f.u = try Array3(f64).init(gpa, 2, n_vars, n_gfpts);
        f.f_comm = try Array3(f64).init(gpa, 2, n_vars, n_gfpts);
        f.norm = try Matrix(f64).init(gpa, n_dims, n_gfpts, null);
        f.d_a = try Matrix(f64).init(gpa, 2, n_gfpts, null);
        f.coord = try Matrix(f64).init(gpa, n_dims, n_gfpts, null);
        f.wave_sp = try gpa.alloc(f64, n_gfpts);
        @memset(f.wave_sp, 0.0);

        if (config.equation.viscous) {
            f.u_comm = try Array3(f64).init(gpa, 2, n_vars, n_gfpts);
            f.u_ldg = try Array3(f64).init(gpa, 2, n_vars, n_gfpts);
            f.du = try Array4(f64).init(gpa, 2, n_dims, n_vars, n_gfpts);
        }

        return f;
    }

    pub fn deinit(f: *Faces) void {
        const gpa = f.gpa;
        f.u.deinit(gpa);
        f.u_comm.deinit(gpa);
        f.u_ldg.deinit(gpa);
        f.f_comm.deinit(gpa);
        f.du.deinit(gpa);
        f.norm.deinit(gpa);
        f.d_a.deinit(gpa);
        f.coord.deinit(gpa);
        gpa.free(f.wave_sp);
    }

    /// Fill the right-hand state of every boundary flux point from its
    /// boundary condition.
    ///
    /// Writes `u` (the ghost state the Rusanov flux sees) and, on viscous runs,
    /// `u_ldg` (the state prescribed to the viscous flux). They differ at a
    /// no-slip wall: the ghost state reflects the velocity so that the Riemann
    /// average is zero, while the prescribed state *is* the wall velocity.
    pub fn applyBcs(f: *Faces) Error!void {
        return switch (f.n_dims) {
            2 => f.applyBcsDim(2),
            3 => f.applyBcsDim(3),
            else => unreachable,
        };
    }

    fn applyBcsDim(f: *Faces, comptime nd: usize) Error!void {
        const equation = f.config.equation.equation;
        const viscous = f.config.equation.viscous;
        const n_vars = f.n_vars;
        const p = f.params;
        const u_fs = p.freestreamState(nd, equation);

        for (f.n_gfpts_int..f.n_gfpts) |gf| {
            const bnd = f.gfpt2bnd[gf - f.n_gfpts_int];
            if (bnd == geo.none) return error.UnmatchedBoundaryFace;
            const bc = f.bc_list[bnd];

            var ul: [nd + 2]f64 = @splat(0.0);
            for (0..n_vars) |n| ul[n] = f.u.get(0, n, gf);

            var norm: [nd]f64 = undefined;
            for (0..nd) |d| norm[d] = f.norm.get(d, gf);

            // Ghost state for the convective flux, and prescribed state for the
            // viscous one. Most conditions set them the same.
            var ur: [nd + 2]f64 = @splat(0.0);
            var ug: [nd + 2]f64 = @splat(0.0);

            switch (bc) {
                .sup_in => {
                    // Farfield / supersonic inflow: impose the freestream
                    ur = u_fs;
                    ug = u_fs;
                },

                .sup_out => {
                    // Supersonic outflow: everything leaves, so extrapolate
                    ur = ul;
                    ug = ul;
                },

                .characteristic => {
                    switch (equation) {
                        // For a scalar equation the characteristic condition is
                        // just upwinding: prescribe on inflow, extrapolate on
                        // outflow. ZEFR only handles the Euler case here.
                        .adv_diff => {
                            var an: f64 = 0.0;
                            for (0..nd) |d| an += p.adv_vel[d] * norm[d];
                            ur[0] = if (an < 0.0) u_fs[0] else ul[0];
                            ug = ur;
                        },
                        .euler_ns => {
                            ur = characteristicState(nd, ul, norm, p);
                            ug = ur;
                        },
                    }
                },

                .slip_wall, .symmetry => {
                    if (viscous) return error.WallConditionMismatch;
                    if (equation == .adv_diff) {
                        // No normal transport: mirror the scalar
                        ur[0] = ul[0];
                        ug = ur;
                    } else {
                        // Reflect the normal momentum, so the Riemann average
                        // has none and the wall is impermeable. Density and
                        // energy are unchanged because |momentum| is.
                        var mom_n: f64 = 0.0;
                        for (0..nd) |d| mom_n += ul[1 + d] * norm[d];

                        ur[0] = ul[0];
                        for (0..nd) |d| ur[1 + d] = ul[1 + d] - 2.0 * mom_n * norm[d];
                        ur[nd + 1] = ul[nd + 1];
                        ug = ur;
                    }
                },

                .isothermal_noslip => {
                    if (!viscous or equation != .euler_ns) return error.WallConditionMismatch;

                    const rho = ul[0];
                    ur[0] = rho;
                    ug[0] = rho;

                    var v_sq: f64 = 0.0;
                    for (0..nd) |d| {
                        const vl = ul[1 + d] / rho;
                        const v = 2.0 * p.vel_wall[d] - vl;
                        ur[1 + d] = rho * v;
                        ug[1 + d] = rho * p.vel_wall[d];
                        v_sq += v * v;
                    }

                    // e_int is fixed by the wall temperature
                    const cv_t = p.r_ref / (p.gamma - 1.0) * p.t_wall;
                    ur[nd + 1] = rho * (cv_t + 0.5 * v_sq);
                    ug[nd + 1] = rho * cv_t;
                },

                .adiabatic_noslip => {
                    if (!viscous or equation != .euler_ns) return error.WallConditionMismatch;

                    const rho = ul[0];
                    ur[0] = rho;
                    ug[0] = rho;

                    // Energy is extrapolated instead of prescribed, with only
                    // the kinetic part adjusted for the new velocity.
                    var v_sq: f64 = 0.0;
                    var vl_sq: f64 = 0.0;
                    var vw_sq: f64 = 0.0;
                    for (0..nd) |d| {
                        const vl = ul[1 + d] / rho;
                        const v = 2.0 * p.vel_wall[d] - vl;
                        ur[1 + d] = rho * v;
                        ug[1 + d] = rho * p.vel_wall[d];
                        v_sq += v * v;
                        vl_sq += vl * vl;
                        vw_sq += p.vel_wall[d] * p.vel_wall[d];
                    }

                    const e_l = ul[nd + 1];
                    ur[nd + 1] = e_l + 0.5 * rho * (v_sq - vl_sq);
                    ug[nd + 1] = e_l + 0.5 * rho * (vw_sq - vl_sq);
                },

                // A periodic face is really an interior face; pairing the two
                // sides needs Flurry-cpp's processPeriodicBoundaries, which is
                // not ported. `.none` means the input file left it unset.
                .periodic, .none => return error.UnsupportedBoundaryCondition,
            }

            for (0..n_vars) |n| f.u.at(1, n, gf).* = ur[n];
            if (viscous) {
                for (0..n_vars) |n| f.u_ldg.at(1, n, gf).* = ug[n];
            }
        }
    }

    /// Boundary conditions on the solution gradient (viscous runs only).
    ///
    /// Only an adiabatic wall constrains the gradient: the wall-normal
    /// temperature gradient must vanish, so there is no heat flux through it.
    /// Every other condition leaves the extrapolated gradient alone.
    pub fn applyBcsGrad(f: *Faces) Error!void {
        if (!f.config.equation.viscous) return;
        return switch (f.n_dims) {
            2 => f.applyBcsGradDim(2),
            3 => f.applyBcsGradDim(3),
            else => unreachable,
        };
    }

    fn applyBcsGradDim(f: *Faces, comptime nd: usize) Error!void {
        const n_vars = f.n_vars;

        for (f.n_gfpts_int..f.n_gfpts) |gf| {
            const bnd = f.gfpt2bnd[gf - f.n_gfpts_int];
            if (bnd == geo.none) return error.UnmatchedBoundaryFace;

            // Default: the right side sees the same gradient as the left
            for (0..nd) |dim| {
                for (0..n_vars) |n| f.du.at(1, dim, n, gf).* = f.du.get(0, dim, n, gf);
            }

            if (f.bc_list[bnd] != .adiabatic_noslip) continue;

            const rho = f.u.get(0, 0, gf);
            const e = f.u.get(0, nd + 1, gf);

            var vel: [nd]f64 = undefined;
            for (0..nd) |d| vel[d] = f.u.get(0, 1 + d, gf) / rho;

            // Remove the wall-normal part of the temperature gradient from the
            // energy gradient. `dt` here is C_v * rho * grad(T).
            var dt: [nd]f64 = undefined;
            for (0..nd) |dim| {
                const drho = f.du.get(0, dim, 0, gf);
                var v_dot_dv: f64 = 0.0;
                for (0..nd) |d| {
                    const dmom = f.du.get(0, dim, 1 + d, gf);
                    v_dot_dv += vel[d] * (dmom - drho * vel[d]) / rho;
                }
                dt[dim] = f.du.get(0, dim, nd + 1, gf) - drho * e / rho - rho * v_dot_dv;
            }

            var dt_dn: f64 = 0.0;
            for (0..nd) |dim| dt_dn += dt[dim] * f.norm.get(dim, gf);

            for (0..nd) |dim| {
                f.du.at(1, dim, nd + 1, gf).* =
                    f.du.get(0, dim, nd + 1, gf) - dt_dn * f.norm.get(dim, gf);
            }
        }
    }

    /// Single-valued interface solution, used by the viscous gradient
    /// correction. LDG with `ldg_b` biasing between the two sides.
    pub fn computeCommonU(f: *Faces) void {
        for (0..f.n_gfpts) |gf| {
            // A boundary prescribes its common state outright; only an interior
            // interface has two sides to bias between.
            const b = if (gf < f.n_gfpts_int) f.config.flux.ldg_b else -0.5;
            for (0..f.n_vars) |n| {
                const ul = f.u_ldg.get(0, n, gf);
                const ur = f.u_ldg.get(1, n, gf);
                const uc = (0.5 + b) * ul + (0.5 - b) * ur;
                // Both sides see the same value, scaled into each one's
                // reference space and signed for its outward normal.
                f.u_comm.at(0, n, gf).* = uc * f.d_a.get(0, gf);
                f.u_comm.at(1, n, gf).* = -uc * f.d_a.get(1, gf);
            }
        }
    }

    /// Common normal flux at every interface, by the configured Riemann solver.
    ///
    /// The result is written for both slots: slot 0 gets `+F dA_0` and slot 1
    /// `-F dA_1`, which is the sign convention `Element.oppDiv_fpts` expects
    /// (it already accounts for each element's own outward normal).
    pub fn computeCommonF(f: *Faces) void {
        switch (f.n_dims) {
            2 => f.rusanov(2),
            3 => f.rusanov(3),
            else => unreachable,
        }
    }

    fn rusanov(f: *Faces, comptime nd: usize) void {
        const equation = f.config.equation.equation;
        const n_vars = f.n_vars;
        const rus_k = f.config.flux.rus_k;

        var ul: [nd + 2]f64 = undefined;
        var ur: [nd + 2]f64 = undefined;
        var norm: [nd]f64 = undefined;

        for (0..f.n_gfpts) |gf| {
            for (0..n_vars) |n| {
                ul[n] = f.u.get(0, n, gf);
                ur[n] = f.u.get(1, n, gf);
            }
            for (0..nd) |d| norm[d] = f.norm.get(d, gf);

            // Normal flux from each side, plus the dissipation scale
            var fnl: [nd + 2]f64 = @splat(0.0);
            var fnr: [nd + 2]f64 = @splat(0.0);
            var eig: f64 = 0.0;

            switch (equation) {
                .adv_diff => {
                    const fl = flux.convAdvDiff(nd, .{ul[0]}, f.params);
                    const fr = flux.convAdvDiff(nd, .{ur[0]}, f.params);
                    for (0..nd) |d| {
                        fnl[0] += fl[0][d] * norm[d];
                        fnr[0] += fr[0][d] * norm[d];
                    }
                    eig = flux.waveSpeed(nd, equation, ul, norm, f.params);
                },
                .euler_ns => {
                    const fl = flux.convEulerNS(nd, ul, f.params);
                    const fr = flux.convEulerNS(nd, ur, f.params);
                    for (0..n_vars) |n| {
                        for (0..nd) |d| {
                            fnl[n] += fl.f[n][d] * norm[d];
                            fnr[n] += fr.f[n][d] * norm[d];
                        }
                    }
                    eig = @max(
                        flux.waveSpeed(nd, equation, ul, norm, f.params),
                        flux.waveSpeed(nd, equation, ur, norm, f.params),
                    );
                },
            }
            f.wave_sp[gf] = eig;

            for (0..n_vars) |n| {
                const fc = 0.5 * (fnl[n] + fnr[n]) -
                    0.5 * eig * (1.0 - rus_k) * (ur[n] - ul[n]);
                f.f_comm.at(0, n, gf).* = fc * f.d_a.get(0, gf);
                f.f_comm.at(1, n, gf).* = -fc * f.d_a.get(1, gf);
            }
        }
    }
};

/// Riemann-invariant ("characteristic") far-field state, after PyFR.
///
/// One incoming and one outgoing invariant are combined into a boundary state
/// that lets waves leave without reflecting, and matches the freestream on
/// inflow. Reduces exactly to the freestream when the interior state already
/// is the freestream.
fn characteristicState(
    comptime nd: usize,
    ul: [nd + 2]f64,
    norm: [nd]f64,
    p: flux.FlowParams,
) [nd + 2]f64 {
    const gam = p.gamma;
    const gm1 = gam - 1.0;

    const rho_l = ul[0];
    var vn_l: f64 = 0.0;
    var vn_r: f64 = 0.0;
    for (0..nd) |d| {
        vn_l += ul[1 + d] / rho_l * norm[d];
        vn_r += p.vel_fs[d] * norm[d];
    }

    const press_l = flux.pressure(nd, ul, gam);
    const press_r = p.p_fs;

    const c_l = @sqrt(gam * press_l / rho_l);
    const c_r = @sqrt(gam * press_r / p.rho_fs);

    // Outgoing invariant, unless the far field is supersonic into the domain
    const r_l = if (@abs(vn_r) >= c_r and vn_l >= 0.0)
        vn_r + 2.0 / gm1 * c_r
    else
        vn_l + 2.0 / gm1 * c_l;

    const r_b = if (@abs(vn_r) >= c_r and vn_l < 0.0)
        vn_l - 2.0 / gm1 * c_l
    else
        vn_r - 2.0 / gm1 * c_r;

    const c_star = 0.25 * gm1 * (r_l - r_b);
    const vn_star = 0.5 * (r_l + r_b);

    // Entropy comes from whichever side the flow is coming from
    var rho_r = c_star * c_star / gam;
    var vel: [nd]f64 = undefined;
    if (vn_l < 0.0) { // inflow
        rho_r *= std.math.pow(f64, p.rho_fs, gam) / press_r;
        for (0..nd) |d| vel[d] = p.vel_fs[d] + (vn_star - vn_r) * norm[d];
    } else { // outflow
        rho_r *= std.math.pow(f64, rho_l, gam) / press_l;
        for (0..nd) |d| vel[d] = ul[1 + d] / rho_l + (vn_star - vn_l) * norm[d];
    }
    rho_r = std.math.pow(f64, rho_r, 1.0 / gm1);

    var ur: [nd + 2]f64 = @splat(0.0);
    ur[0] = rho_r;
    var ke: f64 = 0.0;
    for (0..nd) |d| {
        ur[1 + d] = rho_r * vel[d];
        ke += vel[d] * vel[d];
    }
    const press = rho_r / gam * c_star * c_star;
    ur[nd + 1] = press / gm1 + 0.5 * rho_r * ke;
    return ur;
}

const std = @import("std");

const Config = @import("config.zig").Config;
const cfg = @import("config.zig");
const flux = @import("flux.zig");
const geo = @import("geo.zig");
const Matrix = @import("util/matrix.zig").Matrix;
const Array3 = @import("util/array3.zig").Array3;
const Array4 = @import("util/array4.zig").Array4;

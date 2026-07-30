//! Interface (flux point) coupling: common solution and common normal flux.
//!
//! STATUS: data layout and the numerical flux are implemented; the two pieces
//! that need mesh connectivity we do not have yet are stubbed.
//!
//! Every interior interface appears once here, as a *global flux point* (gfpt)
//! with a left (slot 0) and right (slot 1) side. Elements gather/scatter
//! through `fpt2gfpt` and `fpt2gfpt_slot`, which map an element-local flux
//! point to its gfpt and side.
//!
//! ## What is missing
//!
//! 1. `geo` does not build `fpt2gfpt` / `fpt2gfpt_slot` yet. That needs the
//!    flux points of two adjoining cells matched up in the right rotational
//!    order -- ZEFR's `Elements::setup` plus `FRSolver::orient_fpts`, on top of
//!    the `compareOrientation` logic in Flurry-cpp's `geo.cpp`. Until then
//!    `gatherU`/`scatterF` have nothing to walk and `n_gfpts` is 0.
//!
//! 2. `applyBcs` needs the boundary condition set (characteristic, slip wall,
//!    isothermal/adiabatic no-slip, ...). `geo.bc_type` already says which BC
//!    each boundary face carries, so the missing part is purely the per-BC
//!    state construction.
//!
//! The solver calls both, so the call sites and the data they need are pinned
//! down; filling them in should not change `Solver`.

pub const Error = error{
    OutOfMemory,
    /// A required piece of face connectivity or boundary handling is not ported
    NotImplemented,
};

pub const Faces = struct {
    gpa: std.mem.Allocator,
    config: *const Config,
    params: flux.FlowParams,

    n_dims: usize = 0,
    n_vars: usize = 0,

    /// Global flux points: interior interfaces plus boundary faces
    n_gfpts: usize = 0,
    /// Of those, the ones on a mesh boundary (they occupy the tail)
    n_gfpts_bnd: usize = 0,

    // ---- Interface state, (slot, var, gfpt) ----

    /// Solution on each side of the interface
    u: Array3(f64) = .empty,

    /// Common (single-valued) solution, for the viscous gradient correction
    u_comm: Array3(f64) = .empty,

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
            f.du = try Array4(f64).init(gpa, 2, n_dims, n_vars, n_gfpts);
        }

        return f;
    }

    pub fn deinit(f: *Faces) void {
        const gpa = f.gpa;
        f.u.deinit(gpa);
        f.u_comm.deinit(gpa);
        f.f_comm.deinit(gpa);
        f.du.deinit(gpa);
        f.norm.deinit(gpa);
        f.d_a.deinit(gpa);
        f.coord.deinit(gpa);
        gpa.free(f.wave_sp);
    }

    /// Fill the right-hand state of every boundary flux point from its boundary
    /// condition.
    ///
    /// STUB. Needs the per-BC state construction; `geo.bc_type` already carries
    /// which condition applies to each boundary face. A no-op here leaves the
    /// right state at whatever `gatherU` wrote, which for a boundary face is
    /// zero -- so the residual is only meaningful once this is real.
    pub fn applyBcs(f: *Faces) Error!void {
        if (f.n_gfpts_bnd > 0) return error.NotImplemented;
    }

    /// Same, for the solution gradient (viscous runs only).
    pub fn applyBcsGrad(f: *Faces) Error!void {
        if (f.n_gfpts_bnd > 0) return error.NotImplemented;
    }

    /// Single-valued interface solution, used by the viscous gradient
    /// correction. LDG with `ldg_b` biasing between the two sides.
    pub fn computeCommonU(f: *Faces) void {
        const b = f.config.flux.ldg_b;
        for (0..f.n_gfpts) |gf| {
            for (0..f.n_vars) |n| {
                const ul = f.u.get(0, n, gf);
                const ur = f.u.get(1, n, gf);
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

const std = @import("std");

const Config = @import("config.zig").Config;
const flux = @import("flux.zig");
const Matrix = @import("util/matrix.zig").Matrix;
const Array3 = @import("util/array3.zig").Array3;
const Array4 = @import("util/array4.zig").Array4;

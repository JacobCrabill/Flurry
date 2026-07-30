//! Physical fluxes and the derived flow parameters they need.
//!
//! Every routine here is a pure function of the local state, so they are the
//! easiest part of the solver to test. `n_dims` is a comptime parameter so the
//! loops unroll and the state arrays stay on the stack.
//!
//! Ported from ZEFR's `flux.hpp` and the nondimensionalization in `input.cpp`.

/// Number of conserved variables carried by an equation set.
pub fn nVars(equation: cfg.Equation, n_dims: usize) usize {
    return switch (equation) {
        .adv_diff => 1,
        // density, momentum (n_dims), total energy
        .euler_ns => n_dims + 2,
    };
}

/// Flow parameters derived from the config, in the units the fluxes expect.
///
/// For `euler_ns` these come out of the standard nondimensionalization: the
/// freestream velocity magnitude, density and `rho V^2` are the reference
/// scales, so the freestream state is O(1) and `mu` is effectively 1/Re.
pub const FlowParams = struct {
    gamma: f64 = 1.4,
    prandtl: f64 = 0.72,

    /// Dynamic viscosity (nondimensional unless `disable_nondim`)
    mu: f64 = 0.0,
    /// Reference temperature ratio for Sutherland's law, `T_gas * R / V_fs^2`
    rt: f64 = 1.0,
    /// Sutherland constant over the reference temperature, `S / T_gas`
    c_sth: f64 = 0.0,
    /// Hold `mu` constant instead of applying Sutherland's law
    fix_vis: bool = true,

    /// Advection velocity, `adv_diff` only
    adv_vel: [3]f64 = .{ 0, 0, 0 },
    /// Diffusion coefficient, `adv_diff` only
    diff_coeff: f64 = 0.0,

    // ---- Freestream state, for initialization and boundary conditions ----

    /// Freestream density
    rho_fs: f64 = 1.0,
    /// Freestream pressure
    p_fs: f64 = 1.0,
    /// Freestream velocity vector
    vel_fs: [3]f64 = .{ 0, 0, 0 },

    pub fn fromConfig(config: *const cfg.Config) FlowParams {
        const eq = config.equation;
        const gp = config.gas_properties;
        const fs = config.freestream;
        const n_dims: usize = config.core.n_dims;

        if (eq.equation == .adv_diff) {
            return .{
                .adv_vel = eq.advdiff_A,
                .diff_coeff = eq.advdiff_D,
            };
        }

        // Unit freestream direction
        var norm = fs.norm_fs;
        var mag: f64 = 0.0;
        for (norm) |c| mag += c * c;
        mag = @sqrt(mag);
        if (mag > 0.0) for (&norm) |*c| {
            c.* /= mag;
        };

        var p: FlowParams = .{
            .gamma = gp.gamma,
            .prandtl = gp.prandtl,
            .fix_vis = fs.fix_vis,
            .rho_fs = fs.rho_fs,
            .p_fs = fs.P_fs,
        };

        if (eq.disable_nondim) {
            // Run in the input file's own units: rho, P, Mach, Re and L are
            // given and everything else follows for consistency.
            const v_mag = fs.mach_fs * @sqrt(gp.gamma * fs.P_fs / fs.rho_fs);
            p.mu = fs.rho_fs * v_mag * fs.L_fs / fs.Re_fs;
            p.rt = gp.T_gas * gp.R / (v_mag * v_mag);
            p.c_sth = gp.S / gp.T_gas;
            // Sutherland's law needs a reference temperature scale that the
            // dimensional path does not establish, so viscosity is held fixed.
            p.fix_vis = true;
            for (0..n_dims) |d| p.vel_fs[d] = v_mag * norm[d];
            return p;
        }

        // Dimensional freestream, from the ideal gas law
        const v_mag = fs.mach_fs * @sqrt(gp.gamma * gp.R * fs.T_fs);
        var mu = fs.rho_fs * v_mag * fs.L_fs / fs.Re_fs;
        if (!fs.fix_vis) {
            mu *= std.math.pow(f64, fs.T_fs / gp.T_gas, 1.5) *
                (gp.T_gas + gp.S) / (fs.T_fs + gp.S);
        }

        // Reference scales: rho_fs, V_fs and rho_fs * V_fs^2
        const rho_ref = mu * fs.Re_fs / (v_mag * fs.L_fs);
        const p_ref = rho_ref * v_mag * v_mag;
        const mu_ref = rho_ref * v_mag;

        p.mu = mu / mu_ref;
        p.rt = gp.T_gas * gp.R / (v_mag * v_mag);
        p.c_sth = gp.S / gp.T_gas;
        p.rho_fs = 1.0;
        p.p_fs = (rho_ref * gp.R * fs.T_fs) / p_ref;
        // Velocity is scaled by its own magnitude, so the freestream is a unit
        // vector along `norm_fs`.
        for (0..n_dims) |d| p.vel_fs[d] = norm[d];

        return p;
    }

    /// Freestream state as a conserved-variable vector.
    pub fn freestreamState(p: FlowParams, comptime nd: usize, equation: cfg.Equation) [nd + 2]f64 {
        var u: [nd + 2]f64 = @splat(0.0);
        switch (equation) {
            .adv_diff => u[0] = 1.0,
            .euler_ns => {
                u[0] = p.rho_fs;
                var ke: f64 = 0.0;
                for (0..nd) |d| {
                    u[1 + d] = p.rho_fs * p.vel_fs[d];
                    ke += p.vel_fs[d] * p.vel_fs[d];
                }
                u[nd + 1] = p.p_fs / (p.gamma - 1.0) + 0.5 * p.rho_fs * ke;
            },
        }
        return u;
    }
};

// ---------------------------------------------------------------------------
// Linear advection-diffusion
// ---------------------------------------------------------------------------

/// Convective flux `F = A u`.
pub fn convAdvDiff(comptime nd: usize, u: [1]f64, p: FlowParams) [1][nd]f64 {
    var f: [1][nd]f64 = undefined;
    for (0..nd) |dim| f[0][dim] = p.adv_vel[dim] * u[0];
    return f;
}

/// Add the diffusive flux `-D grad(u)`.
pub fn viscAdvDiffAdd(comptime nd: usize, du: [1][nd]f64, f: *[1][nd]f64, p: FlowParams) void {
    for (0..nd) |dim| f[0][dim] -= p.diff_coeff * du[0][dim];
}

// ---------------------------------------------------------------------------
// Euler / Navier-Stokes
// ---------------------------------------------------------------------------

/// Pressure from the conserved state.
pub fn pressure(comptime nd: usize, u: [nd + 2]f64, gamma: f64) f64 {
    var mom_sq: f64 = 0.0;
    for (0..nd) |d| mom_sq += u[1 + d] * u[1 + d];
    return (gamma - 1.0) * (u[nd + 1] - 0.5 * mom_sq / u[0]);
}

/// Convective (Euler) flux. Also returns the pressure, which the caller
/// usually needs anyway.
pub fn convEulerNS(comptime nd: usize, u: [nd + 2]f64, p: FlowParams) struct {
    f: [nd + 2][nd]f64,
    p: f64,
} {
    const inv_rho = 1.0 / u[0];
    var mom_sq: f64 = 0.0;
    for (0..nd) |d| mom_sq += u[1 + d] * u[1 + d];

    const press = (p.gamma - 1.0) * (u[nd + 1] - 0.5 * mom_sq * inv_rho);
    const h = (u[nd + 1] + press) * inv_rho; // total enthalpy

    var f: [nd + 2][nd]f64 = undefined;
    for (0..nd) |dim| {
        const vel = u[1 + dim] * inv_rho;
        f[0][dim] = u[1 + dim];
        for (0..nd) |d| {
            f[1 + d][dim] = u[1 + d] * vel;
        }
        f[1 + dim][dim] += press;
        f[nd + 1][dim] = u[1 + dim] * h;
    }
    return .{ .f = f, .p = press };
}

/// Add the viscous (Navier-Stokes) flux.
///
/// `du` holds the *conserved*-variable gradients in physical space; the
/// primitive gradients are recovered here.
pub fn viscEulerNSAdd(
    comptime nd: usize,
    u: [nd + 2]f64,
    du: [nd + 2][nd]f64,
    f: *[nd + 2][nd]f64,
    p: FlowParams,
) void {
    const inv_rho = 1.0 / u[0];

    var vel: [nd]f64 = undefined;
    var ke: f64 = 0.0;
    for (0..nd) |d| {
        vel[d] = u[1 + d] * inv_rho;
        ke += vel[d] * vel[d];
    }
    const e_int = u[nd + 1] * inv_rho - 0.5 * ke;

    const mu = if (p.fix_vis) p.mu else sutherland(p, e_int);

    // Velocity gradients: d(rho u)/dx = rho du/dx + u drho/dx
    var dvel: [nd][nd]f64 = undefined;
    for (0..nd) |d| {
        for (0..nd) |dim| {
            dvel[d][dim] = (du[1 + d][dim] - du[0][dim] * vel[d]) * inv_rho;
        }
    }

    // Internal energy gradient, via the kinetic energy
    var de: [nd]f64 = undefined;
    for (0..nd) |dim| {
        var dke: f64 = 0.0;
        for (0..nd) |d| dke += vel[d] * dvel[d][dim];
        dke = 0.5 * ke * du[0][dim] + u[0] * dke;
        de[dim] = (du[nd + 1][dim] - dke - du[0][dim] * e_int) * inv_rho;
    }

    // Newtonian stress tensor with Stokes' hypothesis
    var trace: f64 = 0.0;
    for (0..nd) |d| trace += dvel[d][d];
    const diag = trace / 3.0;

    var tau: [nd][nd]f64 = undefined;
    for (0..nd) |i| {
        for (0..nd) |j| {
            tau[i][j] = mu * (dvel[i][j] + dvel[j][i]);
            if (i == j) tau[i][j] -= 2.0 * mu * diag;
        }
    }

    for (0..nd) |dim| {
        var work: f64 = 0.0;
        for (0..nd) |d| {
            f.*[1 + d][dim] -= tau[d][dim];
            work += vel[d] * tau[d][dim];
        }
        f.*[nd + 1][dim] -= work + (mu / p.prandtl) * p.gamma * de[dim];
    }
}

/// Sutherland's law for the temperature dependence of viscosity.
fn sutherland(p: FlowParams, e_int: f64) f64 {
    const rt_ratio = (p.gamma - 1.0) * e_int / p.rt;
    return p.mu * std.math.pow(f64, rt_ratio, 1.5) *
        (1.0 + p.c_sth) / (rt_ratio + p.c_sth);
}

/// Largest characteristic wave speed normal to `norm`, for the Rusanov
/// dissipation and for the time-step limit.
pub fn waveSpeed(
    comptime nd: usize,
    equation: cfg.Equation,
    u: [nd + 2]f64,
    norm: [nd]f64,
    p: FlowParams,
) f64 {
    switch (equation) {
        .adv_diff => {
            var an: f64 = 0.0;
            for (0..nd) |d| an += p.adv_vel[d] * norm[d];
            return @abs(an);
        },
        .euler_ns => {
            const press = pressure(nd, u, p.gamma);
            const a = @sqrt(p.gamma * press / u[0]);
            var vn: f64 = 0.0;
            for (0..nd) |d| vn += u[1 + d] / u[0] * norm[d];
            return @abs(vn) + a;
        },
    }
}

const std = @import("std");
const cfg = @import("config.zig");

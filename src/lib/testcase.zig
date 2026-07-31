//! Analytic solutions: initial conditions whose exact answer is known for all
//! time, so the scheme's error -- and therefore its order of accuracy -- can be
//! measured rather than inferred.
//!
//! Ported from ZEFR's `compute_U_init` / `compute_U_true` in `funcs.cpp`, with
//! one correction. ZEFR's `compute_U_true` ignores the `t` it is passed, so its
//! error is only meaningful at `t = 0` or after a whole number of periods --
//! which is why its error routine also throws away everything outside a 4x4 box
//! for the Vincent vortex. An isentropic vortex is a steady solution in the
//! frame moving with the free stream, so the exact solution is just the initial
//! condition evaluated upstream: `U(x, t) = U0(x - V t)`. Shifting properly
//! makes the error correct at any time, and the box hack unnecessary.

pub const Error = error{
    /// `test_case.test_case` names a case that is not implemented
    UnsupportedTestCase,
};

/// Conserved state of the exact solution at `(x, y)` and time `t`.
///
/// `bounds` is the periodic domain as `{ {xlo, xhi}, {ylo, yhi} }`: the vortex
/// convects out of one side and back in the other, so the upstream point has to
/// be folded back in.
///
/// Note that a vortex on a finite periodic domain is only an exact solution up
/// to the perturbation it still carries at the boundary. For the Shu vortex on
/// `[-5, 5]^2` that is ~1e-11 in density but ~2e-5 in velocity, which is why
/// density is the variable worth measuring error in -- and why `err_field`
/// defaults to 0.
pub fn exactState(
    tc: TestCase,
    p: flux.FlowParams,
    x: f64,
    y: f64,
    t: f64,
    bounds: [2][2]f64,
) [4]f64 {
    // Velocity of the frame each solution is steady in, so the exact solution
    // at time `t` is the initial condition at `x - vel t`. The sine wave also
    // decays, which is handled in its own branch.
    const vel: [2]f64 = switch (tc) {
        .uniform => .{ 0, 0 },
        .shu_vortex => .{ 1, 1 },
        .vincent_vortex => .{ 0, 1 },
        .sine_wave => .{ p.adv_vel[0], p.adv_vel[1] },
    };
    const xs = wrap(x - vel[0] * t, bounds[0]);
    const ys = wrap(y - vel[1] * t, bounds[1]);

    return switch (tc) {
        .uniform => p.freestreamState(2, .euler_ns),
        .shu_vortex => shuVortex(p.gamma, xs, ys),
        .vincent_vortex => vincentVortex(p.gamma, xs, ys),
        .sine_wave => .{
            @exp(-2.0 * p.diff_coeff * pi * pi * t) * @sin(pi * xs) * @sin(pi * ys),
            0,
            0,
            0,
        },
    };
}

/// Fold `v` back into `[lo, hi)`.
fn wrap(v: f64, bound: [2]f64) f64 {
    const span = bound[1] - bound[0];
    if (span <= 0.0) return v;
    return bound[0] + @mod(v - bound[0], span);
}

/// Shu's isentropic vortex: `G = 5`, core radius 1, on a stream of (1, 1).
///
/// Entropy is uniform and the pressure follows `P = rho^gamma`, so the whole
/// state is fixed by the density -- which is what makes it an exact solution of
/// the Euler equations rather than an approximate one.
fn shuVortex(gamma: f64, x: f64, y: f64) [4]f64 {
    const g = 5.0;
    const radius = 1.0;

    const f = (1.0 - x * x - y * y) / radius;
    const drop = g * g * (gamma - 1.0) / (8.0 * gamma * pi * pi) * @exp(f);

    const rho = std.math.pow(f64, 1.0 - drop, 1.0 / (gamma - 1.0));
    const vx = 1.0 - g * y / (2.0 * pi) * @exp(0.5 * f);
    const vy = 1.0 + g * x / (2.0 * pi) * @exp(0.5 * f);
    const pres = std.math.pow(f64, rho, gamma);

    return conserved(gamma, rho, vx, vy, pres);
}

/// The isentropic vortex of Vincent et al.: `G = 13.5`, core radius 1.5,
/// Mach 0.4, on a stream of (0, 1).
fn vincentVortex(gamma: f64, x: f64, y: f64) [4]f64 {
    const g = 13.5;
    const mach = 0.4;
    const radius = 1.5;

    const omega = g / (2.0 * pi * radius);
    const f = @exp((1.0 - x * x - y * y) / (2.0 * radius * radius));

    const s = omega * mach * radius * f;
    const rho = std.math.pow(f64, 1.0 - 0.5 * (gamma - 1.0) * s * s, 1.0 / (gamma - 1.0));
    const vx = -omega * f * y;
    const vy = 1.0 + omega * f * x;
    // Unit entropy, in the nondimensionalization the Mach number sets
    const pres = std.math.pow(f64, rho, gamma) / (gamma * mach * mach);

    return conserved(gamma, rho, vx, vy, pres);
}

fn conserved(gamma: f64, rho: f64, vx: f64, vy: f64, pres: f64) [4]f64 {
    return .{
        rho,
        rho * vx,
        rho * vy,
        pres / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy),
    };
}

const std = @import("std");
const pi = std.math.pi;

const cfg = @import("config.zig");
const flux = @import("flux.zig");

const TestCase = cfg.TestCase;

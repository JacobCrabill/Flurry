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

/// Conserved state of the exact solution at `x` and time `t`.
///
/// `bounds` is the periodic domain as `{ {xlo, xhi}, ... }` per dimension: the
/// solution convects out of one side and back in the other, so the upstream
/// point has to be folded back in.
///
/// In 3D the two vortices are the same solutions extended along z -- invariant
/// in it, with no velocity through it -- which is an exact solution of the 3D
/// Euler equations, though one that only exercises the third direction's metric
/// terms rather than its physics. The sine wave is genuinely three-dimensional,
/// so it is the one to refine under for a 3D order-of-accuracy study.
///
/// Note that a vortex on a finite periodic domain is only an exact solution up
/// to the perturbation it still carries at the boundary. For the Shu vortex on
/// `[-5, 5]^2` that is ~1e-11 in density but ~2e-5 in velocity, which is why
/// density is the variable worth measuring error in -- and why `err_field`
/// defaults to 0.
pub fn exactState(
    comptime nd: usize,
    tc: TestCase,
    p: flux.FlowParams,
    x: [nd]f64,
    t: f64,
    bounds: [nd][2]f64,
) [nd + 2]f64 {
    // Velocity of the frame each solution is steady in, so the exact solution
    // at time `t` is the initial condition at `x - vel t`. The sine wave also
    // decays, which is handled in its own branch.
    var vel: [nd]f64 = @splat(0.0);
    switch (tc) {
        .uniform => {},
        .shu_vortex => {
            vel[0] = 1;
            vel[1] = 1;
        },
        .vincent_vortex => vel[1] = 1,
        .sine_wave => for (0..nd) |d| {
            vel[d] = p.adv_vel[d];
        },
    }

    var xs: [nd]f64 = undefined;
    for (0..nd) |d| xs[d] = wrap(x[d] - vel[d] * t, bounds[d]);

    return switch (tc) {
        .uniform => p.freestreamState(nd, .euler_ns),
        .shu_vortex => widen(nd, shuVortex(p.gamma, xs[0], xs[1])),
        .vincent_vortex => widen(nd, vincentVortex(p.gamma, xs[0], xs[1])),
        .sine_wave => blk: {
            // The Laplacian of a product of sines picks up one -pi^2 per
            // dimension, so the decay rate follows the dimension count.
            const nd_f: f64 = @floatFromInt(nd);
            var u: [nd + 2]f64 = @splat(0.0);
            var amp = @exp(-nd_f * p.diff_coeff * pi * pi * t);
            for (0..nd) |d| amp *= @sin(pi * xs[d]);
            u[0] = amp;
            break :blk u;
        },
    };
}

/// Widen a 2D conserved state to `nd` dimensions, leaving the extra momentum
/// zero. Energy moves to the end, where it lives for every `nd`.
fn widen(comptime nd: usize, flat: [4]f64) [nd + 2]f64 {
    var u: [nd + 2]f64 = @splat(0.0);
    u[0] = flat[0];
    u[1] = flat[1];
    u[2] = flat[2];
    u[nd + 1] = flat[3];
    return u;
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

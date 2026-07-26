//! Tests for the polynomial basis functions in polynomials.zig.
//!
//! Three layers:
//!
//!   1. A golden regression table generated from the C++ original that
//!      polynomials.zig was ported from (see tools/gen_poly_ref.cpp). This
//!      proves the port is faithful, but by construction cannot catch a bug
//!      that exists in both.
//!   2. Analytic and property tests -- closed forms, orthogonality via Gauss
//!      quadrature, and finite-difference derivative checks. These validate the
//!      mathematics independently of the original.
//!   3. Tests pinning two defects inherited from the C++, each marked FIXME so
//!      that fixing them produces a deliberate, visible test change.

const std = @import("std");
const testing = std.testing;
const p = @import("polynomials.zig");

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Relative comparison that treats NaN as equal to NaN. The reference table
/// contains genuine NaNs (see "Jacobi is singular at a + b == -1" below), and
/// reproducing them exactly is part of matching the original.
fn approxEq(a: f64, b: f64, tol: f64) bool {
    if (std.math.isNan(a) and std.math.isNan(b)) return true;
    if (a == b) return true;
    return @abs(a - b) <= tol * @max(1.0, @max(@abs(a), @abs(b)));
}

fn expectApprox(a: f64, b: f64, tol: f64) !void {
    if (!approxEq(a, b, tol)) {
        std.debug.print("expected {d}, got {d} (tol {d})\n", .{ b, a, tol });
        return error.TestExpectedApproxEq;
    }
}

/// 10-point Gauss-Legendre rule on [-1, 1]; exact through degree 19, which
/// covers every orthogonality check below with room to spare.
const gl_x = [10]f64{
    -0.9739065285171717, -0.8650633666889845, -0.6794095682990244,
    -0.4333953941292472, -0.1488743389816312, 0.1488743389816312,
    0.4333953941292472,  0.6794095682990244,  0.8650633666889845,
    0.9739065285171717,
};
const gl_w = [10]f64{
    0.0666713443086881, 0.1494513491505806, 0.2190863625159820,
    0.2692667193099963, 0.2955242247147529, 0.2955242247147529,
    0.2692667193099963, 0.2190863625159820, 0.1494513491505806,
    0.0666713443086881,
};

fn integrate(comptime f: fn (f64) f64) f64 {
    var sum: f64 = 0.0;
    for (gl_x, gl_w) |x, w| sum += w * f(x);
    return sum;
}

/// Central difference. h = 1e-5 keeps truncation and round-off both near 1e-10
/// for the smooth polynomials here, so 1e-6 is a comfortable tolerance.
const fd_h = 1e-5;

fn centralDiff(comptime f: fn (f64) f64, x: f64) f64 {
    return (f(x + fd_h) - f(x - fd_h)) / (2.0 * fd_h);
}

// ---------------------------------------------------------------------------
// 1. Golden regression table vs the C++ original
// ---------------------------------------------------------------------------

const ref_table = @embedFile("testdata/polynomials_ref.txt");

test "matches C++ reference table" {
    // The table stores %.17g, which round-trips exactly, so the only error
    // source is Zig vs libm differences in pow/gamma/sqrt.
    const tol = 1e-12;

    var checks: usize = 0;
    var fails: usize = 0;

    var lines = std.mem.tokenizeAny(u8, ref_table, "\n");
    while (lines.next()) |line| {
        var tok = std.mem.tokenizeAny(u8, line, " ");
        const tag = tok.next().?;

        var num: [10]f64 = undefined;
        var n: usize = 0;
        while (tok.next()) |t| : (n += 1) {
            num[n] = try std.fmt.parseFloat(f64, t);
        }

        // Recompute each value the generator emitted, in the same order.
        var got: [6]f64 = undefined;
        var want: [6]f64 = undefined;
        var count: usize = 0;

        if (std.mem.eql(u8, tag, "Leg")) {
            const P: u32 = @intFromFloat(num[0]);
            got = .{ p.Legendre(P, num[1]), p.dLegendre(P, num[1]), 0, 0, 0, 0 };
            want = .{ num[2], num[3], 0, 0, 0, 0 };
            count = 2;
        } else if (std.mem.eql(u8, tag, "Jac")) {
            const m: u32 = @intFromFloat(num[0]);
            const xi, const a, const b = .{ num[1], num[2], num[3] };
            got = .{ p.Jacobi(xi, a, b, m), p.dJacobi(xi, a, b, m), 0, 0, 0, 0 };
            want = .{ num[4], num[5], 0, 0, 0, 0 };
            count = 2;
        } else if (std.mem.eql(u8, tag, "Dub2")) {
            const P: u32 = @intFromFloat(num[0]);
            const m: u32 = @intFromFloat(num[1]);
            const xi, const eta = .{ num[2], num[3] };
            got = .{
                p.Dubiner2D(P, xi, eta, m),
                p.dDubiner2D(P, xi, eta, 0, m),
                p.dDubiner2D(P, xi, eta, 1, m),
                0,
                0,
                0,
            };
            want = .{ num[4], num[5], num[6], 0, 0, 0 };
            count = 3;
        } else if (std.mem.eql(u8, tag, "Leg2")) {
            const P: u32 = @intFromFloat(num[0]);
            const m: u32 = @intFromFloat(num[1]);
            got = .{ p.Legendre2D(P, num[2], num[3], m), 0, 0, 0, 0, 0 };
            want = .{ num[4], 0, 0, 0, 0, 0 };
            count = 1;
        } else if (std.mem.eql(u8, tag, "Leg3")) {
            const P: u32 = @intFromFloat(num[0]);
            const m: u32 = @intFromFloat(num[1]);
            got = .{ p.Legendre3D(P, num[2], -0.5, 0.7, m), 0, 0, 0, 0, 0 };
            want = .{ num[3], 0, 0, 0, 0, 0 };
            count = 1;
        } else if (std.mem.eql(u8, tag, "Dub3")) {
            const P: u32 = @intFromFloat(num[0]);
            const m: u32 = @intFromFloat(num[1]);
            const xi, const eta = .{ num[2], num[3] };
            got = .{
                p.Dubiner3D(P, xi, eta, 0.2, m),
                p.dDubiner3D(P, xi, eta, 0.2, 0, m),
                p.dDubiner3D(P, xi, eta, 0.2, 1, m),
                p.dDubiner3D(P, xi, eta, 0.2, 2, m),
                0,
                0,
            };
            want = .{ num[4], num[5], num[6], num[7], 0, 0 };
            count = 4;
        } else if (std.mem.eql(u8, tag, "RT2")) {
            const P: u32 = @intFromFloat(num[0]);
            const m: u32 = @intFromFloat(num[1]);
            const xi, const eta = .{ num[2], num[3] };
            got = .{
                p.RTMonomial2D(P, xi, eta, 0, m),
                p.RTMonomial2D(P, xi, eta, 1, m),
                p.divRTMonomial2D(P, xi, eta, m),
                0,
                0,
                0,
            };
            want = .{ num[4], num[5], num[6], 0, 0, 0 };
            count = 3;
        } else if (std.mem.eql(u8, tag, "RT3")) {
            const P: u32 = @intFromFloat(num[0]);
            const m: u32 = @intFromFloat(num[1]);
            const xi, const eta = .{ num[2], num[3] };
            got = .{
                p.RTMonomial3D(P, xi, eta, 0.25, 0, m),
                p.RTMonomial3D(P, xi, eta, 0.25, 1, m),
                p.RTMonomial3D(P, xi, eta, 0.25, 2, m),
                p.divRTMonomial3D(P, xi, eta, 0.25, m),
                0,
                0,
            };
            want = .{ num[4], num[5], num[6], num[7], 0, 0 };
            count = 4;
        } else {
            std.debug.print("unknown tag '{s}' in reference table\n", .{tag});
            return error.UnknownTag;
        }

        for (got[0..count], want[0..count]) |g, w| {
            checks += 1;
            if (!approxEq(g, w, tol)) {
                fails += 1;
                if (fails <= 20) {
                    std.debug.print("MISMATCH [{s}] got={d} want={d}\n", .{ line, g, w });
                }
            }
        }
    }

    if (fails != 0) {
        std.debug.print("golden table: {d} values checked, {d} mismatches\n", .{ checks, fails });
    }
    // Guards against the table being truncated or a tag silently going unparsed.
    try testing.expect(checks > 20000);
    try testing.expectEqual(@as(usize, 0), fails);
}

// ---------------------------------------------------------------------------
// 2a. Legendre
// ---------------------------------------------------------------------------

fn legendreClosedForm(P: u32, x: f64) f64 {
    return switch (P) {
        0 => 1.0,
        1 => x,
        2 => (3.0 * x * x - 1.0) / 2.0,
        3 => (5.0 * x * x * x - 3.0 * x) / 2.0,
        4 => (35.0 * x * x * x * x - 30.0 * x * x + 3.0) / 8.0,
        else => unreachable,
    };
}

test "Legendre matches closed forms for P <= 4" {
    for ([_]f64{ -1.0, -0.73, -0.25, 0.0, 0.4, 0.9, 1.0 }) |x| {
        for (0..5) |P| {
            const Pu: u32 = @intCast(P);
            try expectApprox(p.Legendre(Pu, x), legendreClosedForm(Pu, x), 1e-14);
        }
    }
}

test "Legendre endpoint values" {
    for (0..8) |P| {
        const Pu: u32 = @intCast(P);
        try expectApprox(p.Legendre(Pu, 1.0), 1.0, 1e-13);
        // P_n(-1) == (-1)^n
        const expected: f64 = if (P % 2 == 0) 1.0 else -1.0;
        try expectApprox(p.Legendre(Pu, -1.0), expected, 1e-13);
    }
}

test "Legendre orthogonality" {
    // int_{-1}^{1} P_i P_j dx == 2/(2i+1) delta_ij
    for (0..6) |i| {
        for (0..6) |j| {
            var sum: f64 = 0.0;
            for (gl_x, gl_w) |x, w| {
                sum += w * p.Legendre(@intCast(i), x) * p.Legendre(@intCast(j), x);
            }
            const fi: f64 = @floatFromInt(i);
            const expected: f64 = if (i == j) 2.0 / (2.0 * fi + 1.0) else 0.0;
            try expectApprox(sum, expected, 1e-12);
        }
    }
}

test "dLegendre matches finite differences" {
    for ([_]f64{ -0.85, -0.4, 0.0, 0.31, 0.77 }) |x| {
        for (0..7) |P| {
            const Pu: u32 = @intCast(P);
            const fd = (p.Legendre(Pu, x + fd_h) - p.Legendre(Pu, x - fd_h)) / (2.0 * fd_h);
            try expectApprox(p.dLegendre(Pu, x), fd, 1e-6);
        }
    }
}

test "dLegendre endpoint value" {
    // P'_n(1) == n(n+1)/2
    for (0..8) |P| {
        const fp: f64 = @floatFromInt(P);
        try expectApprox(p.dLegendre(@intCast(P), 1.0), fp * (fp + 1.0) / 2.0, 1e-12);
    }
}

// ---------------------------------------------------------------------------
// 2b. Jacobi
// ---------------------------------------------------------------------------

test "Jacobi with a = b = 0 reduces to normalized Legendre" {
    // The most valuable check here: two independently-implemented recurrences
    // that must agree, so an error in either one shows up.
    for ([_]f64{ -1.0, -0.62, -0.2, 0.0, 0.45, 0.88, 1.0 }) |x| {
        for (0..7) |n| {
            const fn_: f64 = @floatFromInt(n);
            const expected = p.Legendre(@intCast(n), x) * @sqrt((2.0 * fn_ + 1.0) / 2.0);
            try expectApprox(p.Jacobi(x, 0, 0, @intCast(n)), expected, 1e-11);
        }
    }
}

test "Jacobi is orthonormal under its weight" {
    // int (1-x)^a (1+x)^b J_i J_j dx == delta_ij.
    // With integer a,b the weight is polynomial, so the 10-point rule is exact.
    const cases = [_][2]f64{ .{ 0, 0 }, .{ 1, 0 }, .{ 0, 1 }, .{ 2, 1 }, .{ 3, 2 } };
    for (cases) |c| {
        const a, const b = .{ c[0], c[1] };
        for (0..5) |i| {
            for (0..5) |j| {
                var sum: f64 = 0.0;
                for (gl_x, gl_w) |x, w| {
                    const weight = std.math.pow(f64, 1.0 - x, a) * std.math.pow(f64, 1.0 + x, b);
                    sum += w * weight * p.Jacobi(x, a, b, @intCast(i)) * p.Jacobi(x, a, b, @intCast(j));
                }
                const expected: f64 = if (i == j) 1.0 else 0.0;
                try expectApprox(sum, expected, 1e-10);
            }
        }
    }
}

test "Jacobi mode 0 matches its gamma-function closed form" {
    // Direct coverage of the std::tgamma -> std.math.gamma swap, including
    // fractional arguments where gamma is not just a factorial.
    // Expected values computed independently with Python's math.gamma, not by
    // hand and not from this implementation.
    const cases = [_][3]f64{
        // a, b, expected = sqrt(2^(-a-b-1) * G(a+b+2) / (G(a+1)*G(b+1)))
        .{ 0.0, 0.0, 0.7071067811865476 },
        .{ 1.0, 0.0, 0.7071067811865476 },
        .{ 0.0, 1.0, 0.7071067811865476 },
        .{ 2.0, 0.0, 0.6123724356957945 },
        .{ 1.0, 1.0, 0.8660254037844386 },
        // Fractional and negative a/b: gamma is not a factorial here, so these
        // are the cases that actually exercise the tgamma -> std.math.gamma swap.
        .{ 0.5, 0.5, 0.7978845608028653 },
        .{ -0.5, 0.0, 0.5946035575013606 },
        .{ 1.5, 0.5, 0.7978845608028653 },
    };
    for (cases) |c| {
        const a, const b, const expected = .{ c[0], c[1], c[2] };
        try expectApprox(p.Jacobi(0.37, a, b, 0), expected, 1e-13);
        // Mode 0 is constant in xi.
        try expectApprox(p.Jacobi(-0.8, a, b, 0), expected, 1e-13);
    }
}

test "Jacobi is singular at a + b == -1" {
    // Documents an inherent limitation of the recurrence, not a port bug: for
    // mode >= 2 the (mode-1) term evaluates sqrt(0/0), because
    //   d4 = (mode-1)*((mode-1)+a+b)*... == 0  when mode == 2 and a+b == -1
    //   d5 = ((2*(mode-1))+a+b-1)*((2*(mode-1))+a+b+1) == 0
    // The C++ original does exactly the same. Callers must avoid a + b == -1.
    try testing.expect(std.math.isNan(p.Jacobi(0.3, -0.5, -0.5, 2)));
    try testing.expect(std.math.isNan(p.Jacobi(0.3, -0.5, -0.5, 3)));
    // Modes 0 and 1 do not use the recurrence and stay finite.
    try testing.expect(!std.math.isNan(p.Jacobi(0.3, -0.5, -0.5, 0)));
    try testing.expect(!std.math.isNan(p.Jacobi(0.3, -0.5, -0.5, 1)));
}

test "dJacobi matches finite differences" {
    const cases = [_][2]f64{ .{ 0, 0 }, .{ 1, 0 }, .{ 2, 1 }, .{ 0.5, 1.5 } };
    for (cases) |c| {
        const a, const b = .{ c[0], c[1] };
        for ([_]f64{ -0.7, -0.15, 0.42, 0.81 }) |x| {
            for (0..5) |mode| {
                const m: u32 = @intCast(mode);
                const fd = (p.Jacobi(x + fd_h, a, b, m) - p.Jacobi(x - fd_h, a, b, m)) / (2.0 * fd_h);
                try expectApprox(p.dJacobi(x, a, b, m), fd, 1e-6);
            }
        }
    }
}

test "dJacobi mode 0 is zero" {
    for ([_]f64{ -1.0, -0.3, 0.0, 0.6, 1.0 }) |x| {
        try testing.expectEqual(@as(f64, 0.0), p.dJacobi(x, 0, 0, 0));
        try testing.expectEqual(@as(f64, 0.0), p.dJacobi(x, 2, 1, 0));
    }
}

// ---------------------------------------------------------------------------
// 2c. Dubiner
// ---------------------------------------------------------------------------

test "Dubiner2D derivatives match finite differences" {
    // Interior of the reference triangle only. eta == 1.0 is the collapsed
    // vertex that Dubiner2D special-cases, where the (xi, eta) -> (a, b) map is
    // singular and finite differences are meaningless.
    for (0..4) |P| {
        const Pu: u32 = @intCast(P);
        const nModes = (Pu + 1) * (Pu + 2) / 2;
        for (0..nModes) |mode| {
            const m: u32 = @intCast(mode);
            for ([_]f64{ -0.6, -0.2, 0.1 }) |xi| {
                for ([_]f64{ -0.7, -0.3, 0.2 }) |eta| {
                    if (xi + eta > -0.1) continue; // stay inside the triangle

                    const dxi = (p.Dubiner2D(Pu, xi + fd_h, eta, m) -
                        p.Dubiner2D(Pu, xi - fd_h, eta, m)) / (2.0 * fd_h);
                    try expectApprox(p.dDubiner2D(Pu, xi, eta, 0, m), dxi, 1e-6);

                    const deta = (p.Dubiner2D(Pu, xi, eta + fd_h, m) -
                        p.Dubiner2D(Pu, xi, eta - fd_h, m)) / (2.0 * fd_h);
                    try expectApprox(p.dDubiner2D(Pu, xi, eta, 1, m), deta, 1e-6);
                }
            }
        }
    }
}

test "Dubiner2D mode 0 is constant" {
    // The lowest mode is i = j = 0, so both Jacobi factors are constants and
    // the (1 - b)^0 factor is 1.
    const v = p.Dubiner2D(3, -0.5, -0.25, 0);
    for ([_]f64{ -0.9, -0.4, 0.0 }) |xi| {
        for ([_]f64{ -0.8, -0.3, 0.1 }) |eta| {
            try expectApprox(p.Dubiner2D(3, xi, eta, 0), v, 1e-13);
        }
    }
}

test "Dubiner3D derivatives match finite differences" {
    // Interior of the reference tet only: eta + zeta == 0 and zeta == 1.0 are
    // the collapsed edges/vertex that Dubiner3D special-cases.
    for (0..4) |P| {
        const Pu: u32 = @intCast(P);
        const nModes = (Pu + 1) * (Pu + 2) * (Pu + 3) / 6;
        for (0..nModes) |mode| {
            const m: u32 = @intCast(mode);
            for ([_]f64{ -0.55, -0.2 }) |xi| {
                for ([_]f64{ -0.5, -0.25 }) |eta| {
                    for ([_]f64{ -0.4, -0.15 }) |zeta| {
                        const d0 = (p.Dubiner3D(Pu, xi + fd_h, eta, zeta, m) -
                            p.Dubiner3D(Pu, xi - fd_h, eta, zeta, m)) / (2.0 * fd_h);
                        try expectApprox(p.dDubiner3D(Pu, xi, eta, zeta, 0, m), d0, 1e-6);

                        const d1 = (p.Dubiner3D(Pu, xi, eta + fd_h, zeta, m) -
                            p.Dubiner3D(Pu, xi, eta - fd_h, zeta, m)) / (2.0 * fd_h);
                        try expectApprox(p.dDubiner3D(Pu, xi, eta, zeta, 1, m), d1, 1e-6);

                        const d2 = (p.Dubiner3D(Pu, xi, eta, zeta + fd_h, m) -
                            p.Dubiner3D(Pu, xi, eta, zeta - fd_h, m)) / (2.0 * fd_h);
                        try expectApprox(p.dDubiner3D(Pu, xi, eta, zeta, 2, m), d2, 1e-6);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 2d. RT monomials
// ---------------------------------------------------------------------------

test "RTMonomial2D mode ordering for P = 1" {
    // P=1: nP2Modes = 6. Regular modes pair up as (idx, slot) = (mode/2, mode%2)
    // and idx enumerates (i,j) with i+j <= 1 in the order (0,0), (1,0), (0,1).
    const xi = 0.4;
    const eta = -0.3;
    const expect = [_][3]f64{
        // mode, dim=0 value, dim=1 value
        .{ 0, 1.0, 0.0 }, // idx 0 -> (0,0), slot 0
        .{ 1, 0.0, 1.0 }, // idx 0 -> (0,0), slot 1
        .{ 2, xi, 0.0 }, // idx 1 -> (1,0), slot 0
        .{ 3, 0.0, xi }, // idx 1 -> (1,0), slot 1
        .{ 4, eta, 0.0 }, // idx 2 -> (0,1), slot 0
        .{ 5, 0.0, eta }, // idx 2 -> (0,1), slot 1
    };
    for (expect) |e| {
        const m: u32 = @intFromFloat(e[0]);
        try expectApprox(p.RTMonomial2D(1, xi, eta, 0, m), e[1], 1e-14);
        try expectApprox(p.RTMonomial2D(1, xi, eta, 1, m), e[2], 1e-14);
    }

    // RT modes: idx enumerates (i,j) with i+j == P == 1, i.e. (1,0), (0,1).
    // dim 0 -> xi^(i+1) eta^j, dim 1 -> xi^i eta^(j+1).
    try expectApprox(p.RTMonomial2D(1, xi, eta, 0, 6), xi * xi, 1e-14);
    try expectApprox(p.RTMonomial2D(1, xi, eta, 1, 6), xi * eta, 1e-14);
    try expectApprox(p.RTMonomial2D(1, xi, eta, 0, 7), xi * eta, 1e-14);
    try expectApprox(p.RTMonomial2D(1, xi, eta, 1, 7), eta * eta, 1e-14);
}

test "RTMonomial2D out-of-range mode returns zero" {
    // Past the last RT mode the search loop completes without a match.
    try testing.expectEqual(@as(f64, 0.0), p.RTMonomial2D(1, 0.4, -0.3, 0, 8));
    try testing.expectEqual(@as(f64, 0.0), p.RTMonomial2D(2, 0.4, -0.3, 0, 99));
}

test "RTMonomial3D mode ordering for P = 1" {
    const xi = 0.4;
    const eta = -0.3;
    const zeta = 0.55;
    // Regular modes: (idx, slot) = (mode/3, mode%3); idx enumerates (i,j,k)
    // with i+j+k <= 1 in the order (0,0,0), (1,0,0), (0,1,0), (0,0,1).
    const vals = [_]f64{ 1.0, xi, eta, zeta };
    for (vals, 0..) |v, idx| {
        for (0..3) |slot| {
            const m: u32 = @intCast(idx * 3 + slot);
            for (0..3) |dim| {
                const expected: f64 = if (dim == slot) v else 0.0;
                try expectApprox(p.RTMonomial3D(1, xi, eta, zeta, @intCast(dim), m), expected, 1e-14);
            }
        }
    }
}

test "divRTMonomial2D matches finite-difference divergence" {
    // Skips the i == 0 / j == 0 derivative terms, which are broken -- see the
    // FIXME test below.
    for (1..4) |P| {
        const Pu: u32 = @intCast(P);
        const nModes = (Pu + 1) * (Pu + 2) + (Pu + 1);
        for (0..nModes) |mode| {
            const m: u32 = @intCast(mode);
            for ([_]f64{ -0.65, 0.3, 0.8 }) |xi| {
                for ([_]f64{ -0.45, 0.2, 0.75 }) |eta| {
                    const d0 = (p.RTMonomial2D(Pu, xi + fd_h, eta, 0, m) -
                        p.RTMonomial2D(Pu, xi - fd_h, eta, 0, m)) / (2.0 * fd_h);
                    const d1 = (p.RTMonomial2D(Pu, xi, eta + fd_h, 1, m) -
                        p.RTMonomial2D(Pu, xi, eta - fd_h, 1, m)) / (2.0 * fd_h);
                    const div = p.divRTMonomial2D(Pu, xi, eta, m);
                    if (std.math.isNan(div)) continue; // broken path, pinned below
                    try expectApprox(div, d0 + d1, 1e-6);
                }
            }
        }
    }
}

test "divRTMonomial3D matches finite-difference divergence" {
    for (1..3) |P| {
        const Pu: u32 = @intCast(P);
        const nModes = (Pu + 1) * (Pu + 2) * (Pu + 3) / 2 + (Pu + 1) * (Pu + 2) / 2;
        for (0..nModes) |mode| {
            const m: u32 = @intCast(mode);
            for ([_]f64{ -0.65, 0.35 }) |xi| {
                for ([_]f64{ -0.45, 0.55 }) |eta| {
                    for ([_]f64{ -0.3, 0.7 }) |zeta| {
                        const div = p.divRTMonomial3D(Pu, xi, eta, zeta, m);
                        if (std.math.isNan(div)) continue; // broken path, pinned below

                        const d0 = (p.RTMonomial3D(Pu, xi + fd_h, eta, zeta, 0, m) -
                            p.RTMonomial3D(Pu, xi - fd_h, eta, zeta, 0, m)) / (2.0 * fd_h);
                        const d1 = (p.RTMonomial3D(Pu, xi, eta + fd_h, zeta, 1, m) -
                            p.RTMonomial3D(Pu, xi, eta - fd_h, zeta, 1, m)) / (2.0 * fd_h);
                        const d2 = (p.RTMonomial3D(Pu, xi, eta, zeta + fd_h, 2, m) -
                            p.RTMonomial3D(Pu, xi, eta, zeta - fd_h, 2, m)) / (2.0 * fd_h);

                        try expectApprox(div, d0 + d1 + d2, 1e-6);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 2e. Tensor-product Legendre
// ---------------------------------------------------------------------------

test "Legendre2D is the normalized tensor product, in the documented order" {
    // Re-derives the mode enumeration independently of the implementation.
    for (0..5) |P| {
        const Pu: u32 = @intCast(P);
        var m: u32 = 0;
        for (0..2 * P + 1) |k| {
            for (0..k + 1) |j| {
                const i = k - j;
                if (i > P or j > P) continue;

                const fi: f64 = @floatFromInt(i);
                const fj: f64 = @floatFromInt(j);
                const normCi = @sqrt(2.0 / (2.0 * fi + 1.0));
                const normCj = @sqrt(2.0 / (2.0 * fj + 1.0));

                const xi = 0.35;
                const eta = -0.6;
                const expected = p.Legendre(@intCast(i), xi) *
                    p.Legendre(@intCast(j), eta) / (normCi * normCj);
                try expectApprox(p.Legendre2D(Pu, xi, eta, m), expected, 1e-13);
                m += 1;
            }
        }
        try testing.expectEqual((Pu + 1) * (Pu + 1), m);
    }
}

test "Legendre3D is the normalized tensor product, in the documented order" {
    for (0..4) |P| {
        const Pu: u32 = @intCast(P);
        var m: u32 = 0;
        for (0..3 * P + 1) |l| {
            for (0..l + 1) |k| {
                for (0..l - k + 1) |j| {
                    const i = l - k - j;
                    if (i > P or j > P or k > P) continue;

                    const fi: f64 = @floatFromInt(i);
                    const fj: f64 = @floatFromInt(j);
                    const fk: f64 = @floatFromInt(k);
                    const normCi = @sqrt(2.0 / (2.0 * fi + 1.0));
                    const normCj = @sqrt(2.0 / (2.0 * fj + 1.0));
                    const normCk = @sqrt(2.0 / (2.0 * fk + 1.0));

                    const xi = 0.35;
                    const eta = -0.6;
                    const mu = 0.15;
                    const expected = p.Legendre(@intCast(i), xi) *
                        p.Legendre(@intCast(j), eta) *
                        p.Legendre(@intCast(k), mu) / (normCi * normCj * normCk);
                    try expectApprox(p.Legendre3D(Pu, xi, eta, mu, m), expected, 1e-13);
                    m += 1;
                }
            }
        }
        try testing.expectEqual((Pu + 1) * (Pu + 1) * (Pu + 1), m);
    }
}

// Note: the `mode >= nModes` guards in Dubiner2D/3D, dDubiner2D/3D and
// Legendre2D/3D call @panic, which Zig cannot catch in-process, so those paths
// are deliberately untested.

// ---------------------------------------------------------------------------
// 3. Pinned defects inherited from the C++ original
// ---------------------------------------------------------------------------

test "FIXME: divRTMonomial2D returns NaN for the i == 0 derivative term at 0" {
    // The derivative term is `fi * pow(xi, fi - 1)`. With fi == 0 that is
    // 0 * pow(xi, -1), which is 0 * inf == NaN at xi == 0. The correct value is
    // 0.0 (the derivative of xi^0).
    //
    // The C++ original is wrong here too, but differently: it computes the
    // exponent in unsigned arithmetic, so `i - 1` wraps to 4294967295 and is
    // only then converted to double. That gives 0 * pow(0, 4.29e9) == 0 at
    // xi == 0, but 0 * inf == NaN for every |xi| > 1 -- so the C++ is wrong on a
    // much larger set of inputs. The port subtracts in the float domain because
    // u32 subtraction would panic on overflow in a safe build.
    //
    // Fix: return 0.0 when the monomial degree in the differentiated direction
    // is 0, instead of evaluating pow at a negative exponent.

    // mode 0 -> idx 0, slot 0, (i,j) = (0,0): d/dxi of xi^0 * eta^0.
    try testing.expect(std.math.isNan(p.divRTMonomial2D(1, 0.0, 0.5, 0)));
    // Away from xi == 0 the same mode is accidentally correct.
    try expectApprox(p.divRTMonomial2D(1, 0.3, 0.5, 0), 0.0, 1e-14);

    // mode 1 -> idx 0, slot 1: same defect in the eta direction.
    try testing.expect(std.math.isNan(p.divRTMonomial2D(1, 0.3, 0.0, 1)));
    try expectApprox(p.divRTMonomial2D(1, 0.3, 0.5, 1), 0.0, 1e-14);
}

test "FIXME: divRTMonomial3D returns NaN for the i == 0 derivative term at 0" {
    // Same defect as divRTMonomial2D, in all three slots.
    try testing.expect(std.math.isNan(p.divRTMonomial3D(1, 0.0, 0.5, 0.5, 0)));
    try testing.expect(std.math.isNan(p.divRTMonomial3D(1, 0.5, 0.0, 0.5, 1)));
    try testing.expect(std.math.isNan(p.divRTMonomial3D(1, 0.5, 0.5, 0.0, 2)));

    // Finite away from zero.
    for (0..3) |slot| {
        try expectApprox(p.divRTMonomial3D(1, 0.3, 0.4, 0.5, @intCast(slot)), 0.0, 1e-14);
    }
}

test "FIXME: dDubiner3D returns 0 for an invalid dim" {
    // The C++ original ran off the end of a non-void function here (undefined
    // behaviour); the port returns 0.0 so the behaviour is at least defined.
    // An invalid dim should be a hard error instead of a silent zero.
    try testing.expectEqual(@as(f64, 0.0), p.dDubiner3D(2, -0.5, -0.3, 0.2, 3, 0));
    try testing.expectEqual(@as(f64, 0.0), p.dDubiner3D(2, -0.5, -0.3, 0.2, 7, 1));

    // dim 0..2 do return real values, so the zero above is specific to bad dim.
    try testing.expect(p.dDubiner3D(2, -0.5, -0.3, 0.2, 0, 1) != 0.0);
}

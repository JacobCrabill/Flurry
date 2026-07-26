const std = @import("std");

/// 1D Legendre polynomial
pub fn Legendre(P: u32, xi: f64) f64 {
    if (P == 0) {
        return 1.0;
    } else if (P == 1) {
        return xi;
    }

    const p: f64 = @floatFromInt(P);
    return ((2.0 * p - 1.0) / p) * xi * Legendre(P - 1, xi) - ((p - 1.0) / p) * Legendre(P - 2, xi);
}

/// 1D Legendre polynomial derivative
pub fn dLegendre(P: u32, xi: f64) f64 {
    if (P == 0) {
        return 0.0;
    } else if (P == 1) {
        return 1.0;
    }

    const p: f64 = @floatFromInt(P);
    return ((2.0 * p - 1.0) / p) * (Legendre(P - 1, xi) + xi * dLegendre(P - 1, xi)) - ((p - 1.0) / p) * dLegendre(P - 2, xi);
}

/// 2D Legendre polynomial
pub fn Legendre2D(P: u32, xi: f64, eta: f64, mode: u32) f64 {
    var val: f64 = undefined;
    const nModes: u32 = (P + 1) * (P + 1);
    if (mode >= nModes)
        @panic("ERROR: mode value is too high for given P!");

    var m: u32 = 0;

    for (0..2 * P + 1) |k| {
        for (0..k + 1) |j| {
            const i: u32 = @intCast(k - j);
            if (i <= P and j <= P) { // Order would be (0,2) (1,1) (2,0) ... any hierarchical ordering is fine
                if (m == mode) { // found the correct mode
                    const fi: f64 = @floatFromInt(i);
                    const fj: f64 = @floatFromInt(j);
                    const normCi: f64 = @sqrt(2.0 / (2.0 * fi + 1.0));
                    const normCj: f64 = @sqrt(2.0 / (2.0 * fj + 1.0));
                    val = Legendre(i, xi) * Legendre(@intCast(j), eta) / (normCi * normCj);
                }
                m += 1;
            }
        }
    }

    return val;
}

/// 3D Legendre polynomial
pub fn Legendre3D(P: u32, xi: f64, eta: f64, mu: f64, mode: u32) f64 {
    var val: f64 = undefined;
    const nModes: u32 = (P + 1) * (P + 1) * (P + 1);
    if (mode >= nModes)
        @panic("ERROR: mode value is too high for given P!");

    var m: u32 = 0;
    for (0..3 * P + 1) |l| {
        for (0..l + 1) |k| {
            for (0..l - k + 1) |j| {
                const i: u32 = @intCast(l - k - j);
                if (i <= P and j <= P and k <= P) {
                    if (m == mode) { // found the correct mode
                        const fi: f64 = @floatFromInt(i);
                        const fj: f64 = @floatFromInt(j);
                        const fk: f64 = @floatFromInt(k);
                        const normCi: f64 = @sqrt(2.0 / (2.0 * fi + 1.0));
                        const normCj: f64 = @sqrt(2.0 / (2.0 * fj + 1.0));
                        const normCk: f64 = @sqrt(2.0 / (2.0 * fk + 1.0));
                        val = Legendre(i, xi) * Legendre(@intCast(j), eta) * Legendre(@intCast(k), mu) / (normCi * normCj * normCk);
                    }
                    m += 1;
                }
            }
        }
    }

    return val;
}

/// Jacobi polynomial evaluatoin
pub fn Jacobi(xi: f64, a: f64, b: f64, mode: u32) f64 {
    var val: f64 = 0.0;

    if (mode == 0) {
        var d0: f64 = undefined;
        var d1: f64 = undefined;
        var d2: f64 = undefined;

        d0 = std.math.pow(f64, 2.0, (-a - b - 1));
        d1 = std.math.gamma(f64, a + b + 2);
        d2 = std.math.gamma(f64, a + 1) * std.math.gamma(f64, b + 1);

        val = @sqrt(d0 * (d1 / d2));
    } else if (mode == 1) {
        var d: [6]f64 = undefined;

        d[0] = std.math.pow(f64, 2.0, (-a - b - 1));
        d[1] = std.math.gamma(f64, a + b + 2);
        d[2] = std.math.gamma(f64, a + 1) * std.math.gamma(f64, b + 1);
        d[3] = a + b + 3;
        d[4] = (a + 1) * (b + 1);
        d[5] = (xi * (a + b + 2) + (a - b));

        val = 0.5 * @sqrt(d[0] * (d[1] / d[2])) * @sqrt(d[3] / d[4]) * d[5];
    } else {
        var d: [14]f64 = undefined;
        const m: f64 = @floatFromInt(mode);

        d[0] = m * (m + a + b) * (m + a) * (m + b);
        d[1] = ((2 * m) + a + b - 1) * ((2 * m) + a + b + 1);
        d[2] = (2 * m) + a + b;

        d[3] = (m - 1) * ((m - 1) + a + b) * ((m - 1) + a) * ((m - 1) + b);
        d[4] = ((2 * (m - 1)) + a + b - 1) * ((2 * (m - 1)) + a + b + 1);
        d[5] = (2 * (m - 1)) + a + b;

        d[6] = -((a * a) - (b * b));
        d[7] = ((2 * (m - 1)) + a + b) * ((2 * (m - 1)) + a + b + 2);

        d[8] = (2.0 / d[2]) * @sqrt(d[0] / d[1]);
        d[9] = (2.0 / d[5]) * @sqrt(d[3] / d[4]);
        d[10] = d[6] / d[7];

        d[11] = xi * Jacobi(xi, a, b, mode - 1);
        d[12] = d[9] * Jacobi(xi, a, b, mode - 2);
        d[13] = d[10] * Jacobi(xi, a, b, mode - 1);

        val = (1.0 / d[8]) * (d[11] - d[12] - d[13]);
    }

    return val;
}

/// 1D Jacobi derivative
pub fn dJacobi(xi: f64, a: f64, b: f64, mode: u32) f64 {
    if (mode == 0)
        return 0.0;

    const m: f64 = @floatFromInt(mode);
    return @sqrt(m * (m + a + b + 1)) * Jacobi(xi, a + 1, b + 1, mode - 1);
}

/// 2D Dubiner polynomial
pub fn Dubiner2D(P: u32, xi: f64, eta: f64, mode: u32) f64 {
    var val: f64 = undefined;
    const nModes: u32 = (P + 1) * (P + 2) / 2;
    if (mode >= nModes)
        @panic("ERROR: mode value is too high for given P!");

    const ab: [2]f64 = .{
        if (eta == 1.0) (-1) else ((2 * (1 + xi) / (1 - eta)) - 1),
        eta,
    };

    var m: u32 = 0;
    for (0..P + 1) |k| {
        for (0..k + 1) |j| {
            const i: u32 = @intCast(k - j);
            const fi: f64 = @floatFromInt(i);

            if (m == mode) {
                const j0: f64 = Jacobi(ab[0], 0, 0, i);
                const j1: f64 = Jacobi(ab[1], 2 * fi + 1, 0, @intCast(j));
                val = @sqrt(2.0) * j0 * j1 * std.math.pow(f64, 1 - ab[1], fi);
            }

            m += 1;
        }
    }

    return val;
}

/// 2D Dubiner polynomial derivative (along dimension 'dim')
pub fn dDubiner2D(P: u32, xi: f64, eta: f64, dim: f64, mode: u32) f64 {
    var val: f64 = undefined;
    const nModes: u32 = (P + 1) * (P + 2) / 2;
    if (mode >= nModes)
        @panic("ERROR: mode value is too high for given P!");

    const ab: [2]f64 = .{
        if (eta == 1.0) (-1) else ((2 * (1 + xi) / (1 - eta)) - 1),
        eta,
    };

    var m: u32 = 0;
    for (0..P + 1) |k| {
        for (0..k + 1) |j| {
            const i: u32 = @intCast(k - j);
            const fi: f64 = @floatFromInt(i);

            if (m == mode) {
                if (dim == 0) {
                    const j0: f64 = dJacobi(ab[0], 0, 0, i);
                    const j1: f64 = Jacobi(ab[1], 2 * fi + 1, 0, @intCast(j));
                    if (i == 0) {
                        val = 0.0;
                    } else {
                        val = 2.0 * @sqrt(2.0) * j0 * j1 * std.math.pow(f64, 1 - ab[1], fi - 1);
                    }
                } else if (dim == 1) {
                    const j0: f64 = dJacobi(ab[0], 0, 0, i);
                    const j1: f64 = Jacobi(ab[1], 2 * fi + 1, 0, @intCast(j));
                    const j2: f64 = Jacobi(ab[0], 0, 0, i);
                    const j3: f64 = dJacobi(ab[1], 2 * fi + 1, 0, @intCast(j)) * std.math.pow(f64, 1 - ab[1], fi);
                    const j4: f64 = Jacobi(ab[1], 2 * fi + 1, 0, @intCast(j)) * fi * std.math.pow(f64, 1 - ab[1], fi - 1);

                    if (i == 0) {
                        val = @sqrt(2.0) * j2 * j3;
                    } else {
                        val = @sqrt(2.0) * (j0 * j1 * std.math.pow(f64, 1 - ab[1], fi - 1) * (1 + ab[0]) + j2 * (j3 - j4));
                    }
                }
            }

            m += 1;
        }
    }

    return val;
}

/// Raviart-Thomas 2D polynomial
pub fn RTMonomial2D(P: u32, xi: f64, eta: f64, dim: u32, mode: u32) f64 {
    const nP2Modes: u32 = (P + 1) * (P + 2); // number of regular monomial modes

    if (mode < nP2Modes) {
        // Regular monomial mode
        const idx: u32 = mode / 2;
        const slot: u32 = mode % 2; // which dimension mode is nonzero

        var n: u32 = 0;
        for (0..P + 1) |j| {
            const fj: f64 = @floatFromInt(j);
            for (0..P + 1) |i| {
                const fi: f64 = @floatFromInt(i);
                if (i + j <= P) {
                    if (n == idx) {
                        return if (dim == slot) std.math.pow(f64, xi, fi) * std.math.pow(f64, eta, fj) else 0.0;
                    }
                    n += 1;
                }
            }
        }
    } else {
        // RT mode
        const idx: u32 = mode - nP2Modes;

        var n: u32 = 0;
        for (0..P + 1) |j| {
            const fj: f64 = @floatFromInt(j);
            for (0..P + 1) |i| {
                const fi: f64 = @floatFromInt(i);
                if (i + j == P) {
                    if (n == idx) {
                        return if (dim == 0)
                            std.math.pow(f64, xi, fi + 1) * std.math.pow(f64, eta, fj)
                        else
                            std.math.pow(f64, xi, fi) * std.math.pow(f64, eta, fj + 1);
                    }
                    n += 1;
                }
            }
        }
    }

    return 0.0;
}

pub fn divRTMonomial2D(P: u32, xi: f64, eta: f64, mode: u32) f64 {
    const nP2Modes: u32 = (P + 1) * (P + 2); // number of regular monomial modes

    if (mode < nP2Modes) {
        // Regular monomial mode
        const idx: u32 = mode / 2;
        const slot: u32 = mode % 2; // which dimension mode is nonzero

        var n: u32 = 0;
        for (0..P + 1) |j| {
            const fj: f64 = @floatFromInt(j);
            for (0..P + 1) |i| {
                const fi: f64 = @floatFromInt(i);
                if (i + j <= P) {
                    if (n == idx) {
                        return if (slot == 0)
                            fi * std.math.pow(f64, xi, fi - 1) * std.math.pow(f64, eta, fj)
                        else
                            std.math.pow(f64, xi, fi) * fj * std.math.pow(f64, eta, fj - 1);
                    }
                    n += 1;
                }
            }
        }
    } else {
        // RT mode
        const idx: u32 = mode - nP2Modes;

        var n: u32 = 0;
        for (0..P + 1) |j| {
            const fj: f64 = @floatFromInt(j);
            for (0..P + 1) |i| {
                const fi: f64 = @floatFromInt(i);
                if (i + j == P) {
                    if (n == idx) {
                        return (fi + 1) * std.math.pow(f64, xi, fi) * std.math.pow(f64, eta, fj) +
                            std.math.pow(f64, xi, fi) * (fj + 1) * std.math.pow(f64, eta, fj);
                    }
                    n += 1;
                }
            }
        }
    }

    return 0.0;
}

pub fn Dubiner3D(P: u32, xi: f64, eta: f64, zeta: f64, mode: u32) f64 {
    var val: f64 = undefined;
    const nModes: u32 = (P + 1) * (P + 2) * (P + 3) / 6;
    if (mode >= nModes)
        @panic("ERROR: mode value is too high for given P!");

    const abc: [3]f64 = .{
        if (eta + zeta == 0) (-1) else -2.0 * (1.0 + xi) / (eta + zeta) - 1.0,
        if (zeta == 1) (-1) else 2.0 * (1.0 + eta) / (1.0 - zeta) - 1.0,
        zeta,
    };

    var m: u32 = 0;
    for (0..P + 1) |l| {
        for (0..l + 1) |n| {
            for (0..n + 1) |k| {
                const j: u32 = @intCast(n - k);
                const i: u32 = @intCast(l - j - k);
                const fi: f64 = @floatFromInt(i);
                const fj: f64 = @floatFromInt(j);

                if (m == mode) {
                    const j0: f64 = Jacobi(abc[0], 0, 0, i);
                    const j1: f64 = Jacobi(abc[1], 2 * fi + 1, 0, j);
                    const j2: f64 = Jacobi(abc[2], 2 * fi + 2 * fj + 2, 0, @intCast(k));
                    val = 2.0 * @sqrt(2.0) * j0 * j1 * j2 * std.math.pow(f64, 1 - abc[1], fi) * std.math.pow(f64, 1 - abc[2], fi + fj);
                }

                m += 1;
            }
        }
    }

    return val;
}

pub fn dDubiner3D(P: u32, xi: f64, eta: f64, zeta: f64, dim: f64, mode: u32) f64 {
    const nModes: u32 = (P + 1) * (P + 2) * (P + 3) / 6;
    if (mode >= nModes)
        @panic("ERROR: mode value is too high for given P!");

    const abc: [3]f64 = .{
        if (eta + zeta == 0) (-1) else -2.0 * (1.0 + xi) / (eta + zeta) - 1.0,
        if (zeta == 1) (-1) else 2.0 * (1.0 + eta) / (1.0 - zeta) - 1.0,
        zeta,
    };

    var m: u32 = 0;
    for (0..P + 1) |l| {
        for (0..l + 1) |n| {
            for (0..n + 1) |k| {
                const j: u32 = @intCast(n - k);
                const i: u32 = @intCast(l - j - k);
                const fi: f64 = @floatFromInt(i);
                const fj: f64 = @floatFromInt(j);

                if (m == mode) {
                    const j0: f64 = Jacobi(abc[0], 0, 0, i);
                    const j1: f64 = Jacobi(abc[1], 2 * fi + 1, 0, j);
                    const j2: f64 = Jacobi(abc[2], 2 * fi + 2 * fj + 2, 0, @intCast(k));
                    const dj0: f64 = dJacobi(abc[0], 0, 0, i);
                    const dj1: f64 = dJacobi(abc[1], 2 * fi + 1, 0, j);
                    const dj2: f64 = dJacobi(abc[2], 2 * fi + 2 * fj + 2, 0, @intCast(k));

                    var dxi: f64 = dj0 * j1 * j2;
                    if (i > 0)
                        dxi *= std.math.pow(f64, 0.5 * (1 - abc[1]), fi - 1);
                    if (i + j > 0)
                        dxi *= std.math.pow(f64, 0.5 * (1 - abc[2]), fi + fj - 1);

                    if (dim == 0)
                        return dxi * std.math.pow(f64, 2, 2 * fi + fj + 1.5);

                    var deta: f64 = (0.5 * (1 + abc[0])) * dxi;
                    var tmp: f64 = dj1 * std.math.pow(f64, 0.5 * (1 - abc[1]), fi);

                    if (i > 0)
                        tmp -= 0.5 * fi * j1 * std.math.pow(f64, 0.5 * (1 - abc[1]), fi - 1);
                    if (i + j > 0)
                        tmp *= std.math.pow(f64, 0.5 * (1 - abc[2]), fi + fj - 1);

                    tmp *= j0 * j2;
                    deta += tmp;

                    if (dim == 1)
                        return deta * std.math.pow(f64, 2, 2 * fi + fj + 1.5);

                    var dzeta: f64 = 0.5 * (1 + abc[0]) * dxi + 0.5 * (1 + abc[1]) * tmp;
                    tmp = dj2 * std.math.pow(f64, 0.5 * (1 - abc[2]), fi + fj);

                    if (i + j > 0)
                        tmp -= 0.5 * (fi + fj) * (j2 * std.math.pow(f64, 0.5 * (1 - abc[2]), fi + fj - 1));

                    tmp *= j0 * j1 * std.math.pow(f64, 0.5 * (1 - abc[1]), fi);
                    dzeta += tmp;

                    if (dim == 2)
                        return dzeta * std.math.pow(f64, 2, 2 * fi + fj + 1.5);
                }

                m += 1;
            }
        }
    }

    // The original ran off the end of the function here (no return statement).
    return 0.0;
}

pub fn RTMonomial3D(P: u32, xi: f64, eta: f64, zeta: f64, dim: u32, mode: u32) f64 {
    const nP3Modes: u32 = (P + 1) * (P + 2) * (P + 3) / 2; // number of regular monomial modes

    if (mode < nP3Modes) {
        // Regular monomial mode
        const idx: u32 = mode / 3;
        const slot: u32 = mode % 3; // which dimension mode is nonzero

        var n: u32 = 0;
        for (0..P + 1) |k| {
            const fk: f64 = @floatFromInt(k);
            for (0..P + 1) |j| {
                const fj: f64 = @floatFromInt(j);
                for (0..P + 1) |i| {
                    const fi: f64 = @floatFromInt(i);
                    if (i + j + k <= P) {
                        if (n == idx) {
                            return if (dim == slot)
                                std.math.pow(f64, xi, fi) * std.math.pow(f64, eta, fj) * std.math.pow(f64, zeta, fk)
                            else
                                0.0;
                        }
                        n += 1;
                    }
                }
            }
        }
    } else {
        // RT mode
        const idx: u32 = mode - nP3Modes;

        var n: u32 = 0;
        for (0..P + 1) |k| {
            const fk: f64 = @floatFromInt(k);
            for (0..P + 1) |j| {
                const fj: f64 = @floatFromInt(j);
                for (0..P + 1) |i| {
                    const fi: f64 = @floatFromInt(i);
                    if (i + j + k == P) {
                        if (n == idx) {
                            var val: f64 = undefined;
                            if (dim == 0) {
                                val = std.math.pow(f64, xi, fi + 1) * std.math.pow(f64, eta, fj) * std.math.pow(f64, zeta, fk);
                            } else if (dim == 1) {
                                val = std.math.pow(f64, xi, fi) * std.math.pow(f64, eta, fj + 1) * std.math.pow(f64, zeta, fk);
                            } else if (dim == 2) {
                                val = std.math.pow(f64, xi, fi) * std.math.pow(f64, eta, fj) * std.math.pow(f64, zeta, fk + 1);
                            }

                            return val;
                        }
                        n += 1;
                    }
                }
            }
        }
    }

    return 0.0;
}

pub fn divRTMonomial3D(P: u32, xi: f64, eta: f64, zeta: f64, mode: u32) f64 {
    const nP3Modes: u32 = (P + 1) * (P + 2) * (P + 3) / 2; // number of regular monomial modes

    if (mode < nP3Modes) {
        // Regular monomial mode
        const idx: u32 = mode / 3;
        const slot: u32 = mode % 3; // which dimension mode is nonzero

        var n: u32 = 0;
        for (0..P + 1) |k| {
            const fk: f64 = @floatFromInt(k);
            for (0..P + 1) |j| {
                const fj: f64 = @floatFromInt(j);
                for (0..P + 1) |i| {
                    const fi: f64 = @floatFromInt(i);
                    if (i + j + k <= P) {
                        if (n == idx) {
                            var val: f64 = undefined;
                            if (slot == 0) {
                                val = fi * std.math.pow(f64, xi, fi - 1) * std.math.pow(f64, eta, fj) * std.math.pow(f64, zeta, fk);
                            } else if (slot == 1) {
                                val = std.math.pow(f64, xi, fi) * fj * std.math.pow(f64, eta, fj - 1) * std.math.pow(f64, zeta, fk);
                            } else if (slot == 2) {
                                val = std.math.pow(f64, xi, fi) * std.math.pow(f64, eta, fj) * fk * std.math.pow(f64, zeta, fk - 1);
                            }

                            return val;
                        }
                        n += 1;
                    }
                }
            }
        }
    } else {
        // RT mode
        const idx: u32 = mode - nP3Modes;

        var n: u32 = 0;
        for (0..P + 1) |k| {
            const fk: f64 = @floatFromInt(k);
            for (0..P + 1) |j| {
                const fj: f64 = @floatFromInt(j);
                for (0..P + 1) |i| {
                    const fi: f64 = @floatFromInt(i);
                    if (i + j + k == P) {
                        if (n == idx) {
                            return (fi + 1) * std.math.pow(f64, xi, fi) * std.math.pow(f64, eta, fj) * std.math.pow(f64, zeta, fk) +
                                std.math.pow(f64, xi, fi) * (fj + 1) * std.math.pow(f64, eta, fj) * std.math.pow(f64, zeta, fk) +
                                std.math.pow(f64, xi, fi) * std.math.pow(f64, eta, fj) * (fk + 1) * std.math.pow(f64, zeta, fk);
                        }
                        n += 1;
                    }
                }
            }
        }
    }

    return 0.0;
}

/// VCJH correction functdion (1D)
pub fn Vcjh(xi: f64, mode: u32, order: u32, eta: f64) f64 {
    if (mode == 0) {
        // Left correction function
        return std.math.pow(-1.0, order) / 2.0 * (Legendre(xi, order) -
            (eta * Legendre(xi, order - 1) + Legendre(xi, order + 1)) / (1.0 + eta));
    } else {
        // Right correction function
        return 0.5 * (Legendre(xi, order) + (eta * Legendre(xi, order - 1) + Legendre(xi, order + 1))) /
            (1 + eta);
    }
}

/// Derivative of the VCJH correction functdion (1D)
pub fn dVcjh(in_r: f64, in_mode: u32, in_order: u32, in_eta: f64) f64 {
    if (in_mode == 0) {
        // Left correction function
        if (in_order == 0) {
            return 0.5 * std.math.pow(-1.0, in_order) *
                (dLegendre(in_r, in_order) - ((dLegendre(in_r, in_order + 1)) / (1.0 + in_eta)));
        } else {
            return 0.5 * std.math.pow(-1.0, in_order) * (dLegendre(in_r, in_order) -
                (((in_eta * dLegendre(in_r, in_order - 1)) + dLegendre(in_r, in_order + 1)) /
                    (1.0 + in_eta)));
        }
    } else if (in_mode == 1) {
        // Right correction function
        if (in_order == 0) {
            return 0.5 *
                (dLegendre(in_r, in_order) + ((dLegendre(in_r, in_order + 1)) / (1.0 + in_eta)));
        } else {
            return 0.5 * (dLegendre(in_r, in_order) +
                (((in_eta * dLegendre(in_r, in_order - 1)) + dLegendre(in_r, in_order + 1)) /
                    (1.0 + in_eta)));
        }
    }

    return 0.0;
}

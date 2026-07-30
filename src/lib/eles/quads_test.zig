//! Numerical validation of the quad element's DFR operators.
//!
//! The operators are checked against exact polynomial identities rather than
//! against tabulated reference values, so the tests pin the *mathematics*
//! rather than one particular implementation's output.

const std = @import("std");
const testing = std.testing;

const cfg = @import("../config.zig");
const element = @import("../element.zig");
const Element = element.Element;
const Quad = @import("quads.zig").Quad;
const Matrix = @import("../util/matrix.zig").Matrix;
const points = @import("../points.zig");

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn testConfig(n_qpts_1d: u32) cfg.Config {
    var config: cfg.Config = undefined;
    config.core = .{ .n_dims = 2, .mesh_file = "", .order = 3 };
    config.test_case = .{ .test_case = 0, .err_field = 0, .n_qpts_1d = n_qpts_1d };
    return config;
}

/// A polynomial in (x, y) of degree `deg` in each variable, with fixed
/// pseudo-random coefficients. `eval` and its exact derivatives let us check
/// operators that should be exact on this space.
const Poly = struct {
    deg: usize,
    coef: []const f64,

    fn init(deg: usize, buf: []f64) Poly {
        // Deterministic, spread over a couple of orders of magnitude
        var seed: u64 = 0x9E3779B97F4A7C15;
        for (buf[0 .. (deg + 1) * (deg + 1)]) |*c| {
            seed = seed *% 6364136223846793005 +% 1442695040888963407;
            const r: f64 = @floatFromInt((seed >> 33) % 2000);
            c.* = (r / 1000.0) - 1.0;
        }
        return .{ .deg = deg, .coef = buf[0 .. (deg + 1) * (deg + 1)] };
    }

    fn eval(p: Poly, x: f64, y: f64) f64 {
        var sum: f64 = 0.0;
        for (0..p.deg + 1) |a| {
            for (0..p.deg + 1) |b| {
                sum += p.coef[a * (p.deg + 1) + b] *
                    std.math.pow(f64, x, @floatFromInt(a)) *
                    std.math.pow(f64, y, @floatFromInt(b));
            }
        }
        return sum;
    }

    fn dx(p: Poly, x: f64, y: f64) f64 {
        var sum: f64 = 0.0;
        for (1..p.deg + 1) |a| {
            for (0..p.deg + 1) |b| {
                sum += p.coef[a * (p.deg + 1) + b] * @as(f64, @floatFromInt(a)) *
                    std.math.pow(f64, x, @floatFromInt(a - 1)) *
                    std.math.pow(f64, y, @floatFromInt(b));
            }
        }
        return sum;
    }

    fn dy(p: Poly, x: f64, y: f64) f64 {
        var sum: f64 = 0.0;
        for (0..p.deg + 1) |a| {
            for (1..p.deg + 1) |b| {
                sum += p.coef[a * (p.deg + 1) + b] * @as(f64, @floatFromInt(b)) *
                    std.math.pow(f64, x, @floatFromInt(a)) *
                    std.math.pow(f64, y, @floatFromInt(b - 1));
            }
        }
        return sum;
    }
};

/// Reference-space location of a point-matrix row.
fn loc(mat: *const Matrix(f64), i: usize) [2]f64 {
    return .{ mat.get(i, 0), mat.get(i, 1) };
}

// ---------------------------------------------------------------------------
// 1D point sets (points.zig)
// ---------------------------------------------------------------------------

test "Gauss-Legendre rules integrate exactly to degree 2n-1" {
    const gpa = testing.allocator;

    for (1..11) |n| {
        var pts = try points.gaussLegendrePts(gpa, @intCast(n));
        defer pts.deinit(gpa);
        var wts = try points.gaussLegendreWeights(gpa, @intCast(n));
        defer wts.deinit(gpa);

        try testing.expectEqual(n, pts.data.len);
        try testing.expectEqual(n, wts.data.len);

        // Points are in (-1, 1) and strictly increasing
        for (pts.data) |x| try testing.expect(x > -1.0 and x < 1.0);
        for (1..n) |i| try testing.expect(pts.data[i - 1] < pts.data[i]);

        // integral of x^k over [-1,1] is 0 for odd k, 2/(k+1) for even k
        for (0..2 * n) |k| {
            var sum: f64 = 0.0;
            for (pts.data, wts.data) |x, w| {
                sum += w * std.math.pow(f64, x, @floatFromInt(k));
            }
            const exact: f64 = if (k % 2 == 1) 0.0 else 2.0 / @as(f64, @floatFromInt(k + 1));
            try testing.expectApproxEqAbs(exact, sum, 1e-12);
        }
    }
}

test "Gauss-Legendre points are roots of the Legendre polynomial" {
    const gpa = testing.allocator;
    const poly = @import("../math/polynomials.zig");

    // A direct check on the tabulated values, independent of quadrature. This
    // is what caught a digit typo in the n = 10 table, where the fourth node
    // read 0.4338953... instead of 0.4333953...
    for (1..11) |n| {
        var pts = try points.gaussLegendrePts(gpa, @intCast(n));
        defer pts.deinit(gpa);

        for (pts.data) |x| {
            try testing.expectApproxEqAbs(0.0, poly.Legendre(@intCast(n), x), 1e-14);
        }

        // The nodes are symmetric about the origin
        for (0..n) |i| {
            try testing.expectApproxEqAbs(-pts.data[i], pts.data[n - 1 - i], 1e-15);
        }
    }
}

// ---------------------------------------------------------------------------
// Point layout
// ---------------------------------------------------------------------------

test "quad point layout" {
    const gpa = testing.allocator;
    const config = testConfig(4);

    var quad = Quad.init(gpa, &config, 2, 4);
    defer quad.deinit();
    try quad.ele.setup();

    const ele = &quad.ele;
    try testing.expectEqual(@as(usize, 3), ele.n_spts_1d);
    try testing.expectEqual(@as(usize, 9), ele.n_spts);
    try testing.expectEqual(@as(usize, 12), ele.n_fpts);
    try testing.expectEqual(@as(usize, 9), ele.n_ppts);
    try testing.expectEqual(@as(usize, 16), ele.n_qpts);

    // Solution points are the tensor product of the 1D set, x fastest
    const s = quad.loc_spts_1d;
    for (0..3) |i| {
        for (0..3) |j| {
            const spt = i * 3 + j;
            try testing.expectEqual(s[j], ele.loc_spts.get(spt, 0));
            try testing.expectEqual(s[i], ele.loc_spts.get(spt, 1));
        }
    }

    // The DFR grid brackets the solution points with the face endpoints
    try testing.expectEqual(@as(usize, 5), quad.loc_dfr_1d.len);
    try testing.expectEqual(@as(f64, -1.0), quad.loc_dfr_1d[0]);
    try testing.expectEqual(@as(f64, 1.0), quad.loc_dfr_1d[4]);
    try testing.expectEqualSlices(f64, s, quad.loc_dfr_1d[1..4]);

    // Flux points sit on the right edge, traversed counter-clockwise
    for (0..3) |j| {
        // Bottom: eta = -1, xi increasing
        try testing.expectEqual(@as(f64, -1.0), ele.loc_fpts.get(j, 1));
        try testing.expectEqual(s[j], ele.loc_fpts.get(j, 0));
        // Right: xi = +1, eta increasing
        try testing.expectEqual(@as(f64, 1.0), ele.loc_fpts.get(3 + j, 0));
        try testing.expectEqual(s[j], ele.loc_fpts.get(3 + j, 1));
        // Top: eta = +1, xi decreasing
        try testing.expectEqual(@as(f64, 1.0), ele.loc_fpts.get(6 + j, 1));
        try testing.expectEqual(s[2 - j], ele.loc_fpts.get(6 + j, 0));
        // Left: xi = -1, eta decreasing
        try testing.expectEqual(@as(f64, -1.0), ele.loc_fpts.get(9 + j, 0));
        try testing.expectEqual(s[2 - j], ele.loc_fpts.get(9 + j, 1));
    }

    // Plot points are equispaced and include the corners
    try testing.expectEqual(@as(f64, -1.0), ele.loc_ppts.get(0, 0));
    try testing.expectEqual(@as(f64, -1.0), ele.loc_ppts.get(0, 1));
    try testing.expectEqual(@as(f64, 1.0), ele.loc_ppts.get(8, 0));
    try testing.expectEqual(@as(f64, 1.0), ele.loc_ppts.get(8, 1));

    // Solution weights are a quadrature rule over the bi-unit square
    var area: f64 = 0.0;
    for (ele.weights_spts) |w| area += w;
    try testing.expectApproxEqAbs(@as(f64, 4.0), area, 1e-13);

    var qarea: f64 = 0.0;
    for (ele.weights_qpts) |w| qarea += w;
    try testing.expectApproxEqAbs(@as(f64, 4.0), qarea, 1e-13);

    // Face weights are the 1D rule, integrating a reference edge of length 2
    try testing.expectEqual(@as(usize, 3), ele.weights_fpts.len);
    var edge: f64 = 0.0;
    for (ele.weights_fpts) |w| edge += w;
    try testing.expectApproxEqAbs(@as(f64, 2.0), edge, 1e-13);
}

test "quad parent-space normals" {
    const gpa = testing.allocator;
    const config = testConfig(3);

    var quad = Quad.init(gpa, &config, 1, 4);
    defer quad.deinit();
    try quad.ele.setup();

    const ele = &quad.ele;
    const expect = [4][2]f64{
        .{ 0, -1 }, // bottom
        .{ 1, 0 }, // right
        .{ 0, 1 }, // top
        .{ -1, 0 }, // left
    };
    for (0..ele.n_fpts) |fpt| {
        const n = expect[fpt / ele.n_fpts_per_face];
        try testing.expectEqual(n[0], ele.tnorm.get(fpt, 0));
        try testing.expectEqual(n[1], ele.tnorm.get(fpt, 1));
        try testing.expectEqual(@as(f64, 1.0), ele.tdA[fpt]);
    }

    // Normals point away from the element centre
    for (0..ele.n_fpts) |fpt| {
        var dot: f64 = 0.0;
        for (0..ele.n_dims) |d| dot += ele.tnorm.get(fpt, d) * ele.loc_fpts.get(fpt, d);
        try testing.expect(dot > 0.0);
    }
}

// ---------------------------------------------------------------------------
// Nodal basis
// ---------------------------------------------------------------------------

test "nodal basis is a partition of unity and a Kronecker delta" {
    const gpa = testing.allocator;
    const config = testConfig(3);

    for ([_]u8{ 1, 2, 3, 4 }) |order| {
        var quad = Quad.init(gpa, &config, order, 4);
        defer quad.deinit();
        try quad.ele.setup();
        const ele = &quad.ele;
        const vt = ele.vtable;

        // delta_{spt,k} at the solution points
        for (0..ele.n_spts) |k| {
            const l = loc(&ele.loc_spts, k);
            for (0..ele.n_spts) |spt| {
                const expected: f64 = if (spt == k) 1.0 else 0.0;
                try testing.expectApproxEqAbs(expected, vt.calcNodalBasis(ele, spt, &l), 1e-12);
            }
        }

        // Sums to one anywhere, since constants are in the space
        for ([_][2]f64{ .{ 0.3, -0.7 }, .{ -1.0, 1.0 }, .{ 0.0, 0.0 }, .{ 0.9, 0.9 } }) |l| {
            var sum: f64 = 0.0;
            for (0..ele.n_spts) |spt| sum += vt.calcNodalBasis(ele, spt, &l);
            try testing.expectApproxEqAbs(@as(f64, 1.0), sum, 1e-12);
        }
    }
}

test "oppE extrapolates polynomials to the flux points exactly" {
    const gpa = testing.allocator;
    const config = testConfig(3);

    for ([_]u8{ 1, 2, 3, 4 }) |order| {
        var quad = Quad.init(gpa, &config, order, 4);
        defer quad.deinit();
        try quad.ele.setup();
        const ele = &quad.ele;

        var buf: [64]f64 = undefined;
        const p = Poly.init(order, &buf);

        const u = try gpa.alloc(f64, ele.n_spts);
        defer gpa.free(u);
        for (0..ele.n_spts) |spt| {
            const l = loc(&ele.loc_spts, spt);
            u[spt] = p.eval(l[0], l[1]);
        }

        for (0..ele.n_fpts) |fpt| {
            var sum: f64 = 0.0;
            for (0..ele.n_spts) |spt| sum += ele.oppE.get(fpt, spt) * u[spt];

            const l = loc(&ele.loc_fpts, fpt);
            try testing.expectApproxEqAbs(p.eval(l[0], l[1]), sum, 1e-11);
        }

        // Rows sum to one: extrapolation preserves constants
        for (0..ele.n_fpts) |fpt| {
            var rowsum: f64 = 0.0;
            for (0..ele.n_spts) |spt| rowsum += ele.oppE.get(fpt, spt);
            try testing.expectApproxEqAbs(@as(f64, 1.0), rowsum, 1e-12);
        }
    }
}

test "oppE_ppts and oppE_qpts interpolate polynomials exactly" {
    const gpa = testing.allocator;
    const config = testConfig(4);

    const order: u8 = 3;
    var quad = Quad.init(gpa, &config, order, 4);
    defer quad.deinit();
    try quad.ele.setup();
    const ele = &quad.ele;

    var buf: [64]f64 = undefined;
    const p = Poly.init(order, &buf);

    const u = try gpa.alloc(f64, ele.n_spts);
    defer gpa.free(u);
    for (0..ele.n_spts) |spt| {
        const l = loc(&ele.loc_spts, spt);
        u[spt] = p.eval(l[0], l[1]);
    }

    for (0..ele.n_ppts) |ppt| {
        var sum: f64 = 0.0;
        for (0..ele.n_spts) |spt| sum += ele.oppE_ppts.get(ppt, spt) * u[spt];
        const l = loc(&ele.loc_ppts, ppt);
        try testing.expectApproxEqAbs(p.eval(l[0], l[1]), sum, 1e-11);
    }

    for (0..ele.n_qpts) |qpt| {
        var sum: f64 = 0.0;
        for (0..ele.n_spts) |spt| sum += ele.oppE_qpts.get(qpt, spt) * u[spt];
        const l = loc(&ele.loc_qpts, qpt);
        try testing.expectApproxEqAbs(p.eval(l[0], l[1]), sum, 1e-11);
    }
}

// ---------------------------------------------------------------------------
// DFR operators
// ---------------------------------------------------------------------------

test "oppD_fpts vanishes in the direction tangent to a flux point's face" {
    const gpa = testing.allocator;
    const config = testConfig(3);

    const order: u8 = 3;
    var quad = Quad.init(gpa, &config, order, 4);
    defer quad.deinit();
    try quad.ele.setup();
    const ele = &quad.ele;

    // This is what lets `setupOperators` sum oppDiv_fpts over both dimensions:
    // the DFR endpoint basis function is zero at every solution point, so a
    // bottom/top flux point contributes nothing to d/dxi, and a left/right one
    // nothing to d/deta.
    for (0..ele.n_fpts) |fpt| {
        const face = fpt / ele.n_fpts_per_face;
        const tangential_dim: usize = if (face == 0 or face == 2) 0 else 1;
        for (0..ele.n_spts) |spt| {
            try testing.expectApproxEqAbs(
                @as(f64, 0.0),
                ele.oppD_fpts.get(tangential_dim, spt, fpt),
                1e-11,
            );
        }
    }
}

test "oppD plus oppD_fpts differentiates polynomials exactly" {
    const gpa = testing.allocator;
    const config = testConfig(3);

    for ([_]u8{ 1, 2, 3, 4 }) |order| {
        var quad = Quad.init(gpa, &config, order, 4);
        defer quad.deinit();
        try quad.ele.setup();
        const ele = &quad.ele;

        var buf: [64]f64 = undefined;
        const p = Poly.init(order, &buf);

        const u_spts = try gpa.alloc(f64, ele.n_spts);
        defer gpa.free(u_spts);
        for (0..ele.n_spts) |spt| {
            const l = loc(&ele.loc_spts, spt);
            u_spts[spt] = p.eval(l[0], l[1]);
        }

        const u_fpts = try gpa.alloc(f64, ele.n_fpts);
        defer gpa.free(u_fpts);
        for (0..ele.n_fpts) |fpt| {
            const l = loc(&ele.loc_fpts, fpt);
            u_fpts[fpt] = p.eval(l[0], l[1]);
        }

        // The DFR gradient is the interior term plus the endpoint corrections;
        // together they span the full DFR grid, so a polynomial in the space is
        // differentiated exactly.
        for (0..ele.n_spts) |spt| {
            const l = loc(&ele.loc_spts, spt);

            for (0..ele.n_dims) |dim| {
                var sum: f64 = 0.0;
                for (0..ele.n_spts) |jspt| sum += ele.oppD.get(dim, spt, jspt) * u_spts[jspt];
                for (0..ele.n_fpts) |fpt| sum += ele.oppD_fpts.get(dim, spt, fpt) * u_fpts[fpt];

                const exact = if (dim == 0) p.dx(l[0], l[1]) else p.dy(l[0], l[1]);
                try testing.expectApproxEqAbs(exact, sum, 1e-10);
            }
        }

        // oppDiv is oppD with the dimension index moved, nothing more
        for (0..ele.n_dims) |dim| {
            for (0..ele.n_spts) |i| {
                for (0..ele.n_spts) |j| {
                    try testing.expectEqual(ele.oppD.get(dim, i, j), ele.oppDiv.get(i, dim, j));
                }
            }
        }
    }
}

test "DFR divergence identity holds exactly for polynomial flux fields" {
    const gpa = testing.allocator;
    const config = testConfig(3);

    for ([_]u8{ 1, 2, 3, 4 }) |order| {
        var quad = Quad.init(gpa, &config, order, 4);
        defer quad.deinit();
        try quad.ele.setup();
        const ele = &quad.ele;

        // Two independent polynomials as the flux components
        var buf_x: [64]f64 = undefined;
        var buf_y: [64]f64 = undefined;
        // The y-component evaluates its polynomial with the arguments swapped,
        // so a mistaken component swap in the operators cannot cancel out.
        const fx = Poly.init(order, &buf_x);
        const fy = Poly.init(order, &buf_y);

        const f_spts = try gpa.alloc([2]f64, ele.n_spts);
        defer gpa.free(f_spts);
        for (0..ele.n_spts) |spt| {
            const l = loc(&ele.loc_spts, spt);
            f_spts[spt] = .{ fx.eval(l[0], l[1]), fy.eval(l[1], l[0]) };
        }

        // Normal flux at each flux point: F . n
        const fn_fpts = try gpa.alloc(f64, ele.n_fpts);
        defer gpa.free(fn_fpts);
        for (0..ele.n_fpts) |fpt| {
            const l = loc(&ele.loc_fpts, fpt);
            const f: [2]f64 = .{ fx.eval(l[0], l[1]), fy.eval(l[1], l[0]) };
            fn_fpts[fpt] = f[0] * ele.tnorm.get(fpt, 0) + f[1] * ele.tnorm.get(fpt, 1);
        }

        for (0..ele.n_spts) |spt| {
            const l = loc(&ele.loc_spts, spt);

            var div: f64 = 0.0;
            for (0..ele.n_dims) |dim| {
                for (0..ele.n_spts) |jspt| {
                    div += ele.oppDiv.get(spt, dim, jspt) * f_spts[jspt][dim];
                }
            }
            for (0..ele.n_fpts) |fpt| {
                div += ele.oppDiv_fpts.get(spt, fpt) * fn_fpts[fpt];
            }

            // d/dx[fx(x,y)] + d/dy[fy(y,x)]
            const exact = fx.dx(l[0], l[1]) + fy.dx(l[1], l[0]);
            try testing.expectApproxEqAbs(exact, div, 1e-10);
        }
    }
}

test "oppDiv_fpts sign convention follows the outward normal" {
    const gpa = testing.allocator;
    const config = testConfig(3);

    const order: u8 = 2;
    var quad = Quad.init(gpa, &config, order, 4);
    defer quad.deinit();
    try quad.ele.setup();
    const ele = &quad.ele;

    // Independent check of the sign convention, stated per face rather than by
    // contracting with tnorm the way the implementation does:
    //   bottom (-eta) -> -d/deta,  right (+xi) -> +d/dxi,
    //   top    (+eta) -> +d/deta,  left  (-xi) -> -d/dxi
    for (0..ele.n_fpts) |fpt| {
        const expect: struct { dim: usize, sign: f64 } = switch (fpt / ele.n_fpts_per_face) {
            0 => .{ .dim = 1, .sign = -1.0 },
            1 => .{ .dim = 0, .sign = 1.0 },
            2 => .{ .dim = 1, .sign = 1.0 },
            else => .{ .dim = 0, .sign = -1.0 },
        };
        for (0..ele.n_spts) |spt| {
            try testing.expectApproxEqAbs(
                expect.sign * ele.oppD_fpts.get(expect.dim, spt, fpt),
                ele.oppDiv_fpts.get(spt, fpt),
                1e-12,
            );
        }
    }

    // Correct with F.n = 1 at every flux point. By the divergence theorem the
    // integral of the resulting divergence over the reference square must equal
    // the outward flux through its boundary, i.e. the perimeter. A sign error on
    // any single face would show up here as a cancellation.
    var total: f64 = 0.0;
    for (0..ele.n_spts) |spt| {
        var d: f64 = 0.0;
        for (0..ele.n_fpts) |fpt| d += ele.oppDiv_fpts.get(spt, fpt);
        total += ele.weights_spts[spt] * d;
    }
    try testing.expectApproxEqAbs(@as(f64, 8.0), total, 1e-10);
}

// ---------------------------------------------------------------------------
// Shape (geometry) basis
// ---------------------------------------------------------------------------

/// Gmsh reference coordinates for a linear and a quadratic quad.
const gmsh_quad4 = [4][2]f64{ .{ -1, -1 }, .{ 1, -1 }, .{ 1, 1 }, .{ -1, 1 } };
const gmsh_quad9 = [9][2]f64{
    .{ -1, -1 }, .{ 1, -1 }, .{ 1, 1 }, .{ -1, 1 }, // corners
    .{ 0, -1 }, .{ 1, 0 }, .{ 0, 1 }, .{ -1, 0 }, // edge midpoints
    .{ 0, 0 }, // centre
};

fn checkShapeNodes(gpa: std.mem.Allocator, comptime expect: []const [2]f64) !void {
    const config = testConfig(3);
    var quad = Quad.init(gpa, &config, 2, expect.len);
    defer quad.deinit();
    try quad.ele.setup();
    const ele = &quad.ele;
    const vt = ele.vtable;

    var shape: [64]f64 = undefined;
    var dshape = try Matrix(f64).init(gpa, expect.len, 2, null);
    defer dshape.deinit(gpa);

    // Node ordering: basis `n` is one at node `n` and zero at the others
    for (expect, 0..) |node, n| {
        try vt.calcShape(ele, &node, shape[0..expect.len]);
        for (0..expect.len) |m| {
            const want: f64 = if (m == n) 1.0 else 0.0;
            try testing.expectApproxEqAbs(want, shape[m], 1e-12);
        }
    }

    // Partition of unity, and derivatives summing to zero
    for ([_][2]f64{ .{ 0.3, -0.7 }, .{ -0.2, 0.5 }, .{ 0.0, 0.0 } }) |l| {
        try vt.calcShape(ele, &l, shape[0..expect.len]);
        var sum: f64 = 0.0;
        for (shape[0..expect.len]) |v| sum += v;
        try testing.expectApproxEqAbs(@as(f64, 1.0), sum, 1e-12);

        try vt.calcDShape(ele, &l, &dshape);
        var dsum: [2]f64 = .{ 0.0, 0.0 };
        for (0..expect.len) |m| {
            dsum[0] += dshape.get(m, 0);
            dsum[1] += dshape.get(m, 1);
        }
        try testing.expectApproxEqAbs(@as(f64, 0.0), dsum[0], 1e-12);
        try testing.expectApproxEqAbs(@as(f64, 0.0), dsum[1], 1e-12);

        // The shape map must reproduce the identity: sum of node_i * N_i = loc
        var x: [2]f64 = .{ 0.0, 0.0 };
        for (expect, 0..) |node, m| {
            x[0] += node[0] * shape[m];
            x[1] += node[1] * shape[m];
        }
        try testing.expectApproxEqAbs(l[0], x[0], 1e-12);
        try testing.expectApproxEqAbs(l[1], x[1], 1e-12);
    }
}

test "shape basis matches Gmsh node ordering" {
    try checkShapeNodes(testing.allocator, &gmsh_quad4);
    try checkShapeNodes(testing.allocator, &gmsh_quad9);
}

test "shape basis rejects serendipity quads" {
    const gpa = testing.allocator;
    const config = testConfig(3);

    // 8-node quads have no tensor-product Lagrange basis
    var quad = Quad.init(gpa, &config, 2, 8);
    defer quad.deinit();
    try testing.expectError(error.UnsupportedShapeOrder, quad.ele.setup());
}

// ---------------------------------------------------------------------------
// Faces
// ---------------------------------------------------------------------------

test "face nodes, weights and projection" {
    const gpa = testing.allocator;
    const config = testConfig(3);

    const order: u8 = 3;
    var quad = Quad.init(gpa, &config, order, 4);
    defer quad.deinit();
    try quad.ele.setup();
    const ele = &quad.ele;
    const vt = ele.vtable;

    const pts = try vt.getFaceNodes(ele, gpa, 0, order);
    defer gpa.free(pts);
    const wts = try vt.getFaceWeights(ele, gpa, 0, order);
    defer gpa.free(wts);

    try testing.expectEqual(@as(usize, order + 1), pts.len);
    try testing.expectEqual(@as(usize, order + 1), wts.len);
    var total: f64 = 0.0;
    for (wts) |w| total += w;
    try testing.expectApproxEqAbs(@as(f64, 2.0), total, 1e-13);

    // Projecting a face point must land on that face, and must agree with the
    // flux point ordering: face-local coordinate s maps to the same place the
    // flux points do.
    for (0..ele.n_faces) |face| {
        for (0..ele.n_fpts_per_face) |j| {
            const s = quad.loc_spts_1d[j];
            var ploc: [2]f64 = undefined;
            vt.projectFacePoint(ele, face, &.{s}, &ploc);

            const fpt = face * ele.n_fpts_per_face + j;
            try testing.expectApproxEqAbs(ele.loc_fpts.get(fpt, 0), ploc[0], 1e-14);
            try testing.expectApproxEqAbs(ele.loc_fpts.get(fpt, 1), ploc[1], 1e-14);
        }
    }
}

test "quadrature points are optional" {
    const gpa = testing.allocator;
    // config.Loader.initialize zeroes n_qpts_1d when error_freq == 0, so the
    // quadrature arrays are routinely empty.
    const config = testConfig(0);

    var quad = Quad.init(gpa, &config, 3, 4);
    defer quad.deinit();
    try quad.ele.setup();

    const ele = &quad.ele;
    try testing.expectEqual(@as(usize, 0), ele.n_qpts);
    try testing.expectEqual(@as(usize, 0), ele.weights_qpts.len);
    try testing.expectEqual(@as(usize, 0), ele.oppE_qpts.rows);

    // Everything else must still be built
    try testing.expectEqual(@as(usize, 16), ele.n_spts);
    try testing.expectEqual(@as(usize, 16), ele.oppE.cols);
}

test "operators build across the supported order range" {
    const gpa = testing.allocator;
    const config = testConfig(5);

    // Point sets are tabulated to n = 10, so order 9 is the ceiling
    for (1..10) |order| {
        var quad = Quad.init(gpa, &config, @intCast(order), 4);
        defer quad.deinit();
        try quad.ele.setup();
        const ele = &quad.ele;

        try testing.expectEqual((order + 1) * (order + 1), ele.n_spts);
        try testing.expectEqual((order + 1) * 4, ele.n_fpts);

        // oppE rows still sum to one at every order
        for (0..ele.n_fpts) |fpt| {
            var rowsum: f64 = 0.0;
            for (0..ele.n_spts) |spt| rowsum += ele.oppE.get(fpt, spt);
            try testing.expectApproxEqAbs(@as(f64, 1.0), rowsum, 1e-10);
        }

        // and the divergence correction still integrates to the perimeter
        var total: f64 = 0.0;
        for (0..ele.n_spts) |spt| {
            var d: f64 = 0.0;
            for (0..ele.n_fpts) |fpt| d += ele.oppDiv_fpts.get(spt, fpt);
            total += ele.weights_spts[spt] * d;
        }
        try testing.expectApproxEqAbs(@as(f64, 8.0), total, 1e-8);
    }
}

test "order beyond the tabulated point sets is rejected" {
    const gpa = testing.allocator;
    const config = testConfig(3);

    var quad = Quad.init(gpa, &config, 10, 4);
    defer quad.deinit();
    try testing.expectError(error.UnsupportedOrder, quad.ele.setup());
}

test "Vandermonde matrix of the orthonormal basis" {
    const gpa = testing.allocator;
    const config = testConfig(3);

    const order: u8 = 3;
    var quad = Quad.init(gpa, &config, order, 4);
    defer quad.deinit();
    try quad.ele.setup();
    const ele = &quad.ele;
    const vt = ele.vtable;

    try testing.expectEqual(ele.n_spts, ele.vand.rows);
    try testing.expectEqual(ele.n_spts, ele.vand.cols);

    // vand(i, j) is mode j evaluated at solution point i
    for (0..ele.n_spts) |i| {
        const l = loc(&ele.loc_spts, i);
        for (0..ele.n_spts) |j| {
            try testing.expectEqual(vt.calcOrthonormalBasis(ele, j, &l), ele.vand.get(i, j));
        }
    }

    // The modal basis is orthonormal under the solution-point quadrature, which
    // is exact for these degrees
    for (0..ele.n_spts) |m| {
        for (0..ele.n_spts) |n| {
            var ip: f64 = 0.0;
            for (0..ele.n_spts) |i| {
                ip += ele.weights_spts[i] * ele.vand.get(i, m) * ele.vand.get(i, n);
            }
            const want: f64 = if (m == n) 1.0 else 0.0;
            try testing.expectApproxEqAbs(want, ip, 1e-11);
        }
    }
}

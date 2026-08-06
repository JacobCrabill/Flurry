//! Numerical validation of the hex element's DFR operators, and of the two
//! conventions it shares with the mesh.
//!
//! As in `quads_test.zig`, the operators are checked against exact polynomial
//! identities rather than tabulated reference values. The convention tests are
//! the part with no 2D counterpart: a hex face is a square, so which corner its
//! flux points start at and which way its two axes run are real choices, and
//! `geo.localFaces(.hex)` is where they are recorded.

const std = @import("std");
const testing = std.testing;

const cfg = @import("../config.zig");
const element = @import("../element.zig");
const Element = element.Element;
const Hex = @import("hexes.zig").Hex;
const Matrix = @import("../util/matrix.zig").Matrix;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn testConfig(n_qpts_1d: u32) cfg.Config {
    var config: cfg.Config = undefined;
    config.core = .{ .n_dims = 3, .mesh_file = "", .order = 3 };
    config.test_case = .{ .test_case = .uniform, .err_field = 0, .n_qpts_1d = n_qpts_1d };
    return config;
}

/// A polynomial in (x, y, z) of degree `deg` in each variable, with fixed
/// pseudo-random coefficients.
const Poly = struct {
    deg: usize,
    coef: []const f64,

    fn init(deg: usize, buf: []f64) Poly {
        const n = (deg + 1) * (deg + 1) * (deg + 1);
        // Deterministic, spread over a couple of orders of magnitude
        var seed: u64 = 0x9E3779B97F4A7C15;
        for (buf[0..n]) |*c| {
            seed = seed *% 6364136223846793005 +% 1442695040888963407;
            const r: f64 = @floatFromInt((seed >> 33) % 2000);
            c.* = (r / 1000.0) - 1.0;
        }
        return .{ .deg = deg, .coef = buf[0..n] };
    }

    /// `deriv` is the axis to differentiate along, or null for the plain value.
    fn evalD(p: Poly, l: [3]f64, deriv: ?usize) f64 {
        const n = p.deg + 1;
        var sum: f64 = 0.0;
        for (0..n) |a| {
            for (0..n) |b| {
                for (0..n) |c| {
                    const pow: [3]usize = .{ a, b, c };
                    var term = p.coef[c + n * (b + n * a)];
                    for (0..3) |dim| {
                        const e = pow[dim];
                        if (deriv != null and deriv.? == dim) {
                            if (e == 0) {
                                term = 0.0;
                            } else {
                                term *= @as(f64, @floatFromInt(e)) *
                                    std.math.pow(f64, l[dim], @floatFromInt(e - 1));
                            }
                        } else {
                            term *= std.math.pow(f64, l[dim], @floatFromInt(e));
                        }
                    }
                    sum += term;
                }
            }
        }
        return sum;
    }

    fn eval(p: Poly, l: [3]f64) f64 {
        return p.evalD(l, null);
    }

    fn d(p: Poly, l: [3]f64, dim: usize) f64 {
        return p.evalD(l, dim);
    }
};

/// Reference-space location of a point-matrix row.
fn loc(mat: *const Matrix(f64), i: usize) [3]f64 {
    return .{ mat.get(i, 0), mat.get(i, 1), mat.get(i, 2) };
}

/// Gmsh reference coordinates of a hex's eight corner vertices.
const hex_verts = [8][3]f64{
    .{ -1, -1, -1 }, .{ 1, -1, -1 }, .{ 1, 1, -1 }, .{ -1, 1, -1 },
    .{ -1, -1, 1 },  .{ 1, -1, 1 },  .{ 1, 1, 1 },  .{ -1, 1, 1 },
};

/// `geo.localFaces(.hex)`, restated so that changing either without the other
/// fails here rather than silently mispairing flux points on a 3D mesh.
const local_faces = [6][4]usize{
    .{ 0, 1, 2, 3 }, // bottom, z = -1
    .{ 5, 4, 7, 6 }, // top,    z = +1
    .{ 0, 3, 7, 4 }, // left,   x = -1
    .{ 2, 1, 5, 6 }, // right,  x = +1
    .{ 1, 0, 4, 5 }, // front,  y = -1
    .{ 3, 2, 6, 7 }, // back,   y = +1
};

// ---------------------------------------------------------------------------
// Point layout
// ---------------------------------------------------------------------------

test "hex point layout" {
    const gpa = testing.allocator;
    const config = testConfig(4);

    var hex = Hex.init(gpa, &config, 2, 8);
    defer hex.deinit();
    try hex.ele.setup();

    const ele = &hex.ele;
    try testing.expectEqual(@as(usize, 3), ele.n_spts_1d);
    try testing.expectEqual(@as(usize, 27), ele.n_spts);
    try testing.expectEqual(@as(usize, 54), ele.n_fpts);
    try testing.expectEqual(@as(usize, 9), ele.n_fpts_per_face);
    try testing.expectEqual(@as(usize, 27), ele.n_ppts);
    try testing.expectEqual(@as(usize, 64), ele.n_qpts);

    // Solution points are the triple tensor product of the 1D set, x fastest
    const s = hex.loc_spts_1d;
    for (0..3) |i| {
        for (0..3) |j| {
            for (0..3) |k| {
                const spt = k + 3 * (j + 3 * i);
                try testing.expectEqual(s[k], ele.loc_spts.get(spt, 0));
                try testing.expectEqual(s[j], ele.loc_spts.get(spt, 1));
                try testing.expectEqual(s[i], ele.loc_spts.get(spt, 2));
            }
        }
    }

    // The DFR grid brackets the solution points with the face endpoints
    try testing.expectEqual(@as(usize, 5), hex.loc_dfr_1d.len);
    try testing.expectEqual(@as(f64, -1.0), hex.loc_dfr_1d[0]);
    try testing.expectEqual(@as(f64, 1.0), hex.loc_dfr_1d[4]);
    try testing.expectEqualSlices(f64, s, hex.loc_dfr_1d[1..4]);

    // Plot points are equispaced and include opposite corners
    for (0..3) |d| {
        try testing.expectEqual(@as(f64, -1.0), ele.loc_ppts.get(0, d));
        try testing.expectEqual(@as(f64, 1.0), ele.loc_ppts.get(26, d));
    }

    // Solution weights are a quadrature rule over the bi-unit cube
    var vol: f64 = 0.0;
    for (ele.weights_spts) |w| vol += w;
    try testing.expectApproxEqAbs(@as(f64, 8.0), vol, 1e-13);

    var qvol: f64 = 0.0;
    for (ele.weights_qpts) |w| qvol += w;
    try testing.expectApproxEqAbs(@as(f64, 8.0), qvol, 1e-13);

    // Face weights are the 2D rule, integrating a reference face of area 4
    try testing.expectEqual(@as(usize, 9), ele.weights_fpts.len);
    var area: f64 = 0.0;
    for (ele.weights_fpts) |w| area += w;
    try testing.expectApproxEqAbs(@as(f64, 4.0), area, 1e-13);
}

test "hex parent-space normals" {
    const gpa = testing.allocator;
    const config = testConfig(3);

    var hex = Hex.init(gpa, &config, 1, 8);
    defer hex.deinit();
    try hex.ele.setup();

    const ele = &hex.ele;
    const expect = [6][3]f64{
        .{ 0, 0, -1 }, // bottom
        .{ 0, 0, 1 }, // top
        .{ -1, 0, 0 }, // left
        .{ 1, 0, 0 }, // right
        .{ 0, -1, 0 }, // front
        .{ 0, 1, 0 }, // back
    };
    for (0..ele.n_fpts) |fpt| {
        const n = expect[fpt / ele.n_fpts_per_face];
        for (0..3) |d| try testing.expectEqual(n[d], ele.tnorm.get(fpt, d));
        try testing.expectEqual(@as(f64, 1.0), ele.tdA[fpt]);
    }

    // Normals point away from the element centre
    for (0..ele.n_fpts) |fpt| {
        var dot: f64 = 0.0;
        for (0..ele.n_dims) |d| dot += ele.tnorm.get(fpt, d) * ele.loc_fpts.get(fpt, d);
        try testing.expect(dot > 0.0);
    }
}

test "flux points follow the face vertex order the mesh uses" {
    const gpa = testing.allocator;
    const config = testConfig(3);

    // This is the convention that has no 2D analogue. A quad's face is an
    // interval, and two cells always meet it reversed; a hex's face is a square
    // that two cells can meet in any of eight relative orientations, so the
    // pairing has to know where each side's face starts and which way its axes
    // run. `geo.localFaces(.hex)` is the record of that, and this test is what
    // ties the element's flux points to it.
    for ([_]u8{ 1, 2, 3 }) |order| {
        var hex = Hex.init(gpa, &config, order, 8);
        defer hex.deinit();
        try hex.ele.setup();
        const ele = &hex.ele;

        const n1d = ele.n_spts_1d;
        for (0..6) |face| {
            const fv = local_faces[face];
            const base = face * ele.n_fpts_per_face;

            // Face-local flux points, at the four corners of the face's index
            // square: (fast, slow) = (0,0), (max,0), (0,max)
            const p00 = loc(&ele.loc_fpts, base);
            const pf0 = loc(&ele.loc_fpts, base + (n1d - 1));
            const p0s = loc(&ele.loc_fpts, base + n1d * (n1d - 1));

            // Every flux point on the face lies in the face's plane
            for (0..ele.n_fpts_per_face) |i| {
                const p = loc(&ele.loc_fpts, base + i);
                for (0..3) |d| {
                    // The axis all four face vertices agree on is the normal
                    const v0 = hex_verts[fv[0]][d];
                    const on_plane = for (fv) |v| {
                        if (hex_verts[v][d] != v0) break false;
                    } else true;
                    if (on_plane) try testing.expectEqual(v0, p[d]);
                }
            }

            // The first flux point is the one nearest the face's first vertex
            for (0..3) |d| {
                const toward = hex_verts[fv[0]][d];
                if (toward != 0) try testing.expect(p00[d] * toward > 0.0);
            }

            // The fast index runs from vertex 0 toward vertex 1, the slow index
            // from vertex 0 toward vertex 3.
            for ([_]struct { far: [3]f64, vert: usize }{
                .{ .far = pf0, .vert = fv[1] },
                .{ .far = p0s, .vert = fv[3] },
            }) |case| {
                for (0..3) |d| {
                    const step = case.far[d] - p00[d];
                    const want = hex_verts[case.vert][d] - hex_verts[fv[0]][d];
                    if (want == 0) {
                        try testing.expectApproxEqAbs(@as(f64, 0.0), step, 1e-14);
                    } else {
                        try testing.expect(step * want > 0.0);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Nodal basis
// ---------------------------------------------------------------------------

test "nodal basis is a partition of unity and a Kronecker delta" {
    const gpa = testing.allocator;
    const config = testConfig(3);

    for ([_]u8{ 1, 2, 3 }) |order| {
        var hex = Hex.init(gpa, &config, order, 8);
        defer hex.deinit();
        try hex.ele.setup();
        const ele = &hex.ele;
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
        const probes = [_][3]f64{
            .{ 0.3, -0.7, 0.1 }, .{ -1.0, 1.0, -1.0 },
            .{ 0.0, 0.0, 0.0 },  .{ 0.9, 0.9, -0.4 },
        };
        for (probes) |l| {
            var sum: f64 = 0.0;
            for (0..ele.n_spts) |spt| sum += vt.calcNodalBasis(ele, spt, &l);
            try testing.expectApproxEqAbs(@as(f64, 1.0), sum, 1e-12);
        }
    }
}

test "oppE extrapolates polynomials to the flux points exactly" {
    const gpa = testing.allocator;
    const config = testConfig(3);

    for ([_]u8{ 1, 2, 3 }) |order| {
        var hex = Hex.init(gpa, &config, order, 8);
        defer hex.deinit();
        try hex.ele.setup();
        const ele = &hex.ele;

        var buf: [256]f64 = undefined;
        const p = Poly.init(order, &buf);

        const u = try gpa.alloc(f64, ele.n_spts);
        defer gpa.free(u);
        for (0..ele.n_spts) |spt| u[spt] = p.eval(loc(&ele.loc_spts, spt));

        for (0..ele.n_fpts) |fpt| {
            var sum: f64 = 0.0;
            for (0..ele.n_spts) |spt| sum += ele.oppE.get(fpt, spt) * u[spt];
            try testing.expectApproxEqAbs(p.eval(loc(&ele.loc_fpts, fpt)), sum, 1e-11);
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
    var hex = Hex.init(gpa, &config, order, 8);
    defer hex.deinit();
    try hex.ele.setup();
    const ele = &hex.ele;

    var buf: [256]f64 = undefined;
    const p = Poly.init(order, &buf);

    const u = try gpa.alloc(f64, ele.n_spts);
    defer gpa.free(u);
    for (0..ele.n_spts) |spt| u[spt] = p.eval(loc(&ele.loc_spts, spt));

    for (0..ele.n_ppts) |ppt| {
        var sum: f64 = 0.0;
        for (0..ele.n_spts) |spt| sum += ele.oppE_ppts.get(ppt, spt) * u[spt];
        try testing.expectApproxEqAbs(p.eval(loc(&ele.loc_ppts, ppt)), sum, 1e-11);
    }

    for (0..ele.n_qpts) |qpt| {
        var sum: f64 = 0.0;
        for (0..ele.n_spts) |spt| sum += ele.oppE_qpts.get(qpt, spt) * u[spt];
        try testing.expectApproxEqAbs(p.eval(loc(&ele.loc_qpts, qpt)), sum, 1e-11);
    }
}

// ---------------------------------------------------------------------------
// Derivative and divergence operators
// ---------------------------------------------------------------------------

test "oppD_fpts vanishes in the directions tangent to a flux point's face" {
    const gpa = testing.allocator;
    const config = testConfig(3);

    var hex = Hex.init(gpa, &config, 3, 8);
    defer hex.deinit();
    try hex.ele.setup();
    const ele = &hex.ele;

    // What lets `setupOperators` sum oppDiv_fpts over all three dimensions: the
    // DFR endpoint basis function is zero at every solution point, so a flux
    // point contributes only to the derivative along its own face normal.
    const normal_dim = [6]usize{ 2, 2, 0, 0, 1, 1 };
    for (0..ele.n_fpts) |fpt| {
        const nd = normal_dim[fpt / ele.n_fpts_per_face];
        for (0..3) |dim| {
            if (dim == nd) continue;
            for (0..ele.n_spts) |spt| {
                try testing.expectApproxEqAbs(0.0, ele.oppD_fpts.get(dim, spt, fpt), 1e-11);
            }
        }
    }
}

test "oppD plus oppD_fpts differentiates polynomials exactly" {
    const gpa = testing.allocator;
    const config = testConfig(3);

    for ([_]u8{ 1, 2, 3 }) |order| {
        var hex = Hex.init(gpa, &config, order, 8);
        defer hex.deinit();
        try hex.ele.setup();
        const ele = &hex.ele;

        var buf: [256]f64 = undefined;
        const p = Poly.init(order, &buf);

        const u_spts = try gpa.alloc(f64, ele.n_spts);
        defer gpa.free(u_spts);
        for (0..ele.n_spts) |spt| u_spts[spt] = p.eval(loc(&ele.loc_spts, spt));

        const u_fpts = try gpa.alloc(f64, ele.n_fpts);
        defer gpa.free(u_fpts);
        for (0..ele.n_fpts) |fpt| u_fpts[fpt] = p.eval(loc(&ele.loc_fpts, fpt));

        // The DFR gradient is the interior term plus the endpoint corrections;
        // together they span the full DFR grid, so a polynomial in the space is
        // differentiated exactly.
        for (0..ele.n_spts) |spt| {
            const l = loc(&ele.loc_spts, spt);
            for (0..ele.n_dims) |dim| {
                var sum: f64 = 0.0;
                for (0..ele.n_spts) |jspt| sum += ele.oppD.get(dim, spt, jspt) * u_spts[jspt];
                for (0..ele.n_fpts) |fpt| sum += ele.oppD_fpts.get(dim, spt, fpt) * u_fpts[fpt];
                try testing.expectApproxEqAbs(p.d(l, dim), sum, 1e-10);
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

    for ([_]u8{ 1, 2, 3 }) |order| {
        var hex = Hex.init(gpa, &config, order, 8);
        defer hex.deinit();
        try hex.ele.setup();
        const ele = &hex.ele;

        // Three independent polynomials as the flux components. Each evaluates
        // its own with the arguments rotated, so a mistaken component swap in
        // the operators cannot cancel out.
        var buf_x: [256]f64 = undefined;
        var buf_y: [256]f64 = undefined;
        var buf_z: [256]f64 = undefined;
        const fx = Poly.init(order, &buf_x);
        const fy = Poly.init(order, &buf_y);
        const fz = Poly.init(order, &buf_z);

        // F(l) = (fx(x,y,z), fy(y,z,x), fz(z,x,y))
        const flux = struct {
            fn at(px: Poly, py: Poly, pz: Poly, l: [3]f64) [3]f64 {
                return .{
                    px.eval(.{ l[0], l[1], l[2] }),
                    py.eval(.{ l[1], l[2], l[0] }),
                    pz.eval(.{ l[2], l[0], l[1] }),
                };
            }
        }.at;

        const f_spts = try gpa.alloc([3]f64, ele.n_spts);
        defer gpa.free(f_spts);
        for (0..ele.n_spts) |spt| f_spts[spt] = flux(fx, fy, fz, loc(&ele.loc_spts, spt));

        // Normal flux at each flux point: F . n
        const fn_fpts = try gpa.alloc(f64, ele.n_fpts);
        defer gpa.free(fn_fpts);
        for (0..ele.n_fpts) |fpt| {
            const f = flux(fx, fy, fz, loc(&ele.loc_fpts, fpt));
            var dot: f64 = 0.0;
            for (0..3) |d| dot += f[d] * ele.tnorm.get(fpt, d);
            fn_fpts[fpt] = dot;
        }

        for (0..ele.n_spts) |spt| {
            const l = loc(&ele.loc_spts, spt);

            var div: f64 = 0.0;
            for (0..ele.n_dims) |dim| {
                for (0..ele.n_spts) |jspt| {
                    div += ele.oppDiv.get(spt, dim, jspt) * f_spts[jspt][dim];
                }
            }
            for (0..ele.n_fpts) |fpt| div += ele.oppDiv_fpts.get(spt, fpt) * fn_fpts[fpt];

            // Each component is differentiated along its own first argument
            const exact = fx.d(.{ l[0], l[1], l[2] }, 0) +
                fy.d(.{ l[1], l[2], l[0] }, 0) +
                fz.d(.{ l[2], l[0], l[1] }, 0);
            try testing.expectApproxEqAbs(exact, div, 1e-10);
        }
    }
}

test "oppDiv_fpts sign convention follows the outward normal" {
    const gpa = testing.allocator;
    const config = testConfig(3);

    var hex = Hex.init(gpa, &config, 2, 8);
    defer hex.deinit();
    try hex.ele.setup();
    const ele = &hex.ele;

    // Stated per face rather than by contracting with tnorm the way the
    // implementation does: bottom (-z) -> -d/dz, top (+z) -> +d/dz, and so on.
    const expect = [6]struct { dim: usize, sign: f64 }{
        .{ .dim = 2, .sign = -1.0 }, // bottom
        .{ .dim = 2, .sign = 1.0 }, // top
        .{ .dim = 0, .sign = -1.0 }, // left
        .{ .dim = 0, .sign = 1.0 }, // right
        .{ .dim = 1, .sign = -1.0 }, // front
        .{ .dim = 1, .sign = 1.0 }, // back
    };
    for (0..ele.n_fpts) |fpt| {
        const e = expect[fpt / ele.n_fpts_per_face];
        for (0..ele.n_spts) |spt| {
            try testing.expectApproxEqAbs(
                e.sign * ele.oppD_fpts.get(e.dim, spt, fpt),
                ele.oppDiv_fpts.get(spt, fpt),
                1e-12,
            );
        }
    }

    // Correct with F.n = 1 at every flux point. By the divergence theorem the
    // integral of the resulting divergence over the reference cube must equal
    // the outward flux through its boundary, i.e. the surface area, 6 * 4 = 24.
    // A sign error on any single face would show up here as a cancellation.
    var total: f64 = 0.0;
    for (0..ele.n_spts) |spt| {
        var d: f64 = 0.0;
        for (0..ele.n_fpts) |fpt| d += ele.oppDiv_fpts.get(spt, fpt);
        total += ele.weights_spts[spt] * d;
    }
    try testing.expectApproxEqAbs(@as(f64, 24.0), total, 1e-10);
}

// ---------------------------------------------------------------------------
// Shape (geometry) basis
// ---------------------------------------------------------------------------

/// Gmsh reference coordinates for the 27-node quadratic hex: eight corners,
/// then the twelve edge midpoints in Gmsh's edge order, then the six face
/// centres in Gmsh's face order, then the volume centre.
const gmsh_hex27 = [27][3]f64{
    .{ -1, -1, -1 }, .{ 1, -1, -1 }, .{ 1, 1, -1 }, .{ -1, 1, -1 },
    .{ -1, -1, 1 },  .{ 1, -1, 1 },  .{ 1, 1, 1 },  .{ -1, 1, 1 },
    .{ 0, -1, -1 }, // 8:  edge (0,1)
    .{ -1, 0, -1 }, // 9:  edge (0,3)
    .{ -1, -1, 0 }, // 10: edge (0,4)
    .{ 1, 0, -1 }, // 11: edge (1,2)
    .{ 1, -1, 0 }, // 12: edge (1,5)
    .{ 0, 1, -1 }, // 13: edge (2,3)
    .{ 1, 1, 0 }, // 14: edge (2,6)
    .{ -1, 1, 0 }, // 15: edge (3,7)
    .{ 0, -1, 1 }, // 16: edge (4,5)
    .{ -1, 0, 1 }, // 17: edge (4,7)
    .{ 1, 0, 1 }, // 18: edge (5,6)
    .{ 0, 1, 1 }, // 19: edge (6,7)
    .{ 0, 0, -1 }, // 20: face (0,3,2,1)
    .{ 0, -1, 0 }, // 21: face (0,1,5,4)
    .{ -1, 0, 0 }, // 22: face (0,4,7,3)
    .{ 1, 0, 0 }, // 23: face (1,2,6,5)
    .{ 0, 1, 0 }, // 24: face (2,3,7,6)
    .{ 0, 0, 1 }, // 25: face (4,5,6,7)
    .{ 0, 0, 0 }, // 26: volume centre
};

/// The shape basis must be the Kronecker delta at the node positions the mesh
/// file uses, which is what makes the mapping interpolate the mesh.
fn checkShapeNodes(gpa: std.mem.Allocator, comptime expect: []const [3]f64) !void {
    const config = testConfig(3);

    var hex = Hex.init(gpa, &config, 2, expect.len);
    defer hex.deinit();
    try hex.ele.setup();
    const ele = &hex.ele;

    const shape = try gpa.alloc(f64, expect.len);
    defer gpa.free(shape);

    for (expect, 0..) |x, node| {
        try ele.vtable.calcShape(ele, &x, shape);
        for (shape, 0..) |v, n| {
            const expected: f64 = if (n == node) 1.0 else 0.0;
            try testing.expectApproxEqAbs(expected, v, 1e-12);
        }
    }

    // And a partition of unity away from the nodes
    var dshape = try Matrix(f64).init(gpa, expect.len, 3, null);
    defer dshape.deinit(gpa);
    for ([_][3]f64{ .{ 0.2, -0.4, 0.7 }, .{ -0.9, 0.1, -0.3 } }) |x| {
        try ele.vtable.calcShape(ele, &x, shape);
        var sum: f64 = 0.0;
        for (shape) |v| sum += v;
        try testing.expectApproxEqAbs(@as(f64, 1.0), sum, 1e-12);

        // The derivatives of a partition of unity cancel
        try ele.vtable.calcDShape(ele, &x, &dshape);
        for (0..3) |d| {
            var dsum: f64 = 0.0;
            for (0..expect.len) |n| dsum += dshape.get(n, d);
            try testing.expectApproxEqAbs(@as(f64, 0.0), dsum, 1e-12);
        }
    }
}

test "shape basis matches Gmsh node ordering" {
    const gpa = testing.allocator;
    try checkShapeNodes(gpa, &hex_verts);
    try checkShapeNodes(gpa, &gmsh_hex27);
}

test "shape basis rejects hexes with no tensor-product layout" {
    const gpa = testing.allocator;
    const config = testConfig(3);

    // 20-node serendipity hexes have no Lagrange basis; nor does any node
    // count that is not a perfect cube, or one too small to be a hex.
    for ([_]usize{ 0, 1, 20, 26, 30 }) |n_nodes| {
        var hex = Hex.init(gpa, &config, 2, n_nodes);
        defer hex.deinit();
        try testing.expectError(error.UnsupportedShapeOrder, hex.ele.setup());
    }
}

/// Gmsh's edge list for a hexahedron, in the order its edge nodes are written.
const gmsh_hex_edges = [12][2]usize{
    .{ 0, 1 }, .{ 0, 3 }, .{ 0, 4 }, .{ 1, 2 }, .{ 1, 5 }, .{ 2, 3 },
    .{ 2, 6 }, .{ 3, 7 }, .{ 4, 5 }, .{ 4, 7 }, .{ 5, 6 }, .{ 6, 7 },
};

/// Gmsh's face list, in the order its face nodes are written. Each is oriented
/// outward, and that orientation is what fixes the in-plane numbering of the
/// face's interior nodes -- three of the six come out transposed or reflected
/// relative to the cube axes because of it.
const gmsh_hex_faces = [6][4]usize{
    .{ 0, 3, 2, 1 }, .{ 0, 1, 5, 4 }, .{ 0, 4, 7, 3 },
    .{ 1, 2, 6, 5 }, .{ 2, 3, 7, 6 }, .{ 4, 5, 6, 7 },
};

/// Recover `structured (i, j, k) -> Gmsh node` from the shape basis alone, by
/// evaluating it at each structured node position and taking the one function
/// that is 1 there. Going through the public interface means the check below is
/// independent of how the map is built.
fn recoverNodeMap(gpa: std.mem.Allocator, ele: *const Element, n_side: usize) ![]usize {
    const out = try gpa.alloc(usize, ele.n_nodes);
    errdefer gpa.free(out);

    const shape = try gpa.alloc(f64, ele.n_nodes);
    defer gpa.free(shape);

    const step = 2.0 / @as(f64, @floatFromInt(n_side - 1));
    for (0..n_side) |k| {
        for (0..n_side) |j| {
            for (0..n_side) |i| {
                const x: [3]f64 = .{
                    -1.0 + step * @as(f64, @floatFromInt(i)),
                    -1.0 + step * @as(f64, @floatFromInt(j)),
                    -1.0 + step * @as(f64, @floatFromInt(k)),
                };
                try ele.vtable.calcShape(ele, &x, shape);

                var found: ?usize = null;
                for (shape, 0..) |v, n| {
                    if (@abs(v - 1.0) < 1e-10) {
                        try testing.expect(found == null);
                        found = n;
                    } else {
                        try testing.expectApproxEqAbs(@as(f64, 0.0), v, 1e-10);
                    }
                }
                out[i + n_side * (j + n_side * k)] = found orelse return error.NoNodeThere;
            }
        }
    }
    return out;
}

test "Gmsh hex node map follows the corner, edge and face recursion" {
    const gpa = testing.allocator;
    const config = testConfig(3);

    // n_side 2 has no interior nodes at all, 3 has one per edge and face, 4 is
    // the first with a face ring of four, and 5 the first with a ring plus a
    // centre. Between them they exercise every branch of the recursion.
    for ([_]usize{ 2, 3, 4, 5 }) |n_side| {
        const n_nodes = n_side * n_side * n_side;
        var hex = Hex.init(gpa, &config, 2, n_nodes);
        defer hex.deinit();
        try hex.ele.setup();

        const ijk2gmsh = try recoverNodeMap(gpa, &hex.ele, n_side);
        defer gpa.free(ijk2gmsh);

        const hi = n_side - 1;
        const idx = struct {
            fn at(n: usize, c: [3]usize) usize {
                return c[0] + n * (c[1] + n * c[2]);
            }
        }.at;

        // Structured coordinates of each Gmsh corner
        var corner: [8][3]usize = undefined;
        for (hex_verts, 0..) |v, n| {
            for (0..3) |d| corner[n][d] = if (v[d] > 0) hi else 0;
            try testing.expectEqual(n, ijk2gmsh[idx(n_side, corner[n])]);
        }

        const n_int = n_side - 2; // interior points along one edge
        if (n_int == 0) continue;

        // Edges: node block `e` runs from its first vertex to its second
        for (gmsh_hex_edges, 0..) |edge, e| {
            const a = corner[edge[0]];
            const b = corner[edge[1]];
            for (0..n_int) |t| {
                var c: [3]usize = undefined;
                for (0..3) |d| {
                    // Exactly one axis differs, and the walk is along it
                    c[d] = if (a[d] == b[d]) a[d] else if (b[d] > a[d]) t + 1 else hi - t - 1;
                }
                const want = 8 + e * n_int + t;
                try testing.expectEqual(want, ijk2gmsh[idx(n_side, c)]);
            }
        }

        // Faces: the first ring of a face block sits one step inside the face's
        // own four vertices, taken in the face's own order. This is what pins
        // the transposed and reflected faces.
        //
        // A face with one interior node per edge has no ring, only that single
        // centre node -- `gmsh_hex27` covers that case exactly.
        const face_base = 8 + 12 * n_int;
        if (n_int < 2) continue;
        for (gmsh_hex_faces, 0..) |face, f| {
            const p0 = corner[face[0]];
            for (face, 0..) |v, corner_of_face| {
                var c: [3]usize = undefined;
                for (0..3) |d| {
                    // Step one node inward from this vertex, in the two
                    // in-plane directions; the normal direction is unchanged.
                    const at_hi = corner[v][d] == hi;
                    const fixed = corner[face[0]][d] == corner[face[1]][d] and
                        corner[face[0]][d] == corner[face[2]][d];
                    c[d] = if (fixed) p0[d] else if (at_hi) hi - 1 else 1;
                }
                const want = face_base + f * n_int * n_int + corner_of_face;
                try testing.expectEqual(want, ijk2gmsh[idx(n_side, c)]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Faces
// ---------------------------------------------------------------------------

test "face nodes, weights and projection" {
    const gpa = testing.allocator;
    const config = testConfig(3);

    const order: u8 = 3;
    var hex = Hex.init(gpa, &config, order, 8);
    defer hex.deinit();
    try hex.ele.setup();
    const ele = &hex.ele;
    const vt = ele.vtable;

    const n = @as(usize, order) + 1;
    const pts = try vt.getFaceNodes(ele, gpa, 0, order);
    defer gpa.free(pts);
    const wts = try vt.getFaceWeights(ele, gpa, 0, order);
    defer gpa.free(wts);

    // A face rule is the 2D tensor product, `(n^2, 2)` points against `n^2`
    // weights covering a reference square of area 4.
    try testing.expectEqual(n * n * 2, pts.len);
    try testing.expectEqual(n * n, wts.len);
    var area: f64 = 0.0;
    for (wts) |w| area += w;
    try testing.expectApproxEqAbs(@as(f64, 4.0), area, 1e-13);

    // Projecting a face's own flux point locations back up must land exactly on
    // the element's flux points -- the continuous statement of the reversal
    // `setLocs` applies to three of the faces.
    for (0..6) |face| {
        for (0..ele.n_fpts_per_face) |i| {
            const l: [2]f64 = .{
                hex.loc_spts_1d[i % n],
                hex.loc_spts_1d[i / n],
            };
            var ploc: [3]f64 = undefined;
            vt.projectFacePoint(ele, face, &l, &ploc);

            const fpt = face * ele.n_fpts_per_face + i;
            for (0..3) |d| {
                try testing.expectApproxEqAbs(ele.loc_fpts.get(fpt, d), ploc[d], 1e-14);
            }
        }
    }

    // The face-local nodal basis is the delta at those same points
    for (0..ele.n_fpts_per_face) |pt| {
        for (0..ele.n_fpts_per_face) |q| {
            const l: [2]f64 = .{ hex.loc_spts_1d[q % n], hex.loc_spts_1d[q / n] };
            const expected: f64 = if (pt == q) 1.0 else 0.0;
            try testing.expectApproxEqAbs(expected, vt.calcNodalFaceBasis(ele, 0, pt, &l), 1e-12);
        }
    }
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

test "operators build across the supported order range" {
    const gpa = testing.allocator;
    const config = testConfig(3);

    for ([_]u8{ 1, 2, 3, 4 }) |order| {
        var hex = Hex.init(gpa, &config, order, 8);
        defer hex.deinit();
        try hex.ele.setup();

        const ele = &hex.ele;
        const n1d = @as(usize, order) + 1;
        try testing.expectEqual(n1d * n1d * n1d, ele.n_spts);
        try testing.expectEqual(6 * n1d * n1d, ele.n_fpts);
        try testing.expectEqual(ele.n_fpts, ele.oppE.rows);
        try testing.expectEqual(ele.n_spts, ele.oppE.cols);
    }
}

test "quadrature points are optional" {
    const gpa = testing.allocator;
    const config = testConfig(0);

    // `error_freq = 0` zeroes n_qpts_1d, and a run that never measures error
    // should not pay for the points.
    var hex = Hex.init(gpa, &config, 3, 8);
    defer hex.deinit();
    try hex.ele.setup();

    try testing.expectEqual(@as(usize, 0), hex.ele.n_qpts);
}

test "Vandermonde matrix of the orthonormal basis" {
    const gpa = testing.allocator;
    const config = testConfig(3);

    var hex = Hex.init(gpa, &config, 2, 8);
    defer hex.deinit();
    try hex.ele.setup();
    const ele = &hex.ele;

    try testing.expectEqual(ele.n_spts, ele.vand.rows);
    try testing.expectEqual(ele.n_spts, ele.vand.cols);

    // Row i is the modal basis evaluated at solution point i
    for (0..ele.n_spts) |i| {
        const l = loc(&ele.loc_spts, i);
        for (0..ele.n_spts) |j| {
            const v = ele.vtable.calcOrthonormalBasis(ele, j, &l);
            try testing.expectApproxEqAbs(v, ele.vand.get(i, j), 1e-14);
        }
    }
}

//! 2D Quadrilateral elements
//!
//! Tensor products of 1D Lagrange bases. Two 1D point sets are in play:
//!
//!   loc_spts_1d   the n_spts_1d solution points (Gauss-Legendre)
//!   loc_dfr_1d    the same points with -1 and +1 prepended/appended
//!
//! The solution is interpolated on `loc_spts_1d`; the flux is interpolated on
//! `loc_dfr_1d`, which is what makes this Direct Flux Reconstruction rather
//! than plain FR. Because `loc_dfr_1d` contains every solution point, the two
//! endpoint basis functions vanish at all of them -- the property that lets
//! `Element.setupOperators` sum the correction over all dimensions.
//!
//! Ported from ZEFR's `quads.cpp`.

pub const Quad = struct {
    ele: Element,

    /// Solution points in 1D (Gauss-Legendre), length `ele.n_spts_1d`
    loc_spts_1d: []f64 = &.{},

    /// `loc_spts_1d` bracketed by -1 and +1; the DFR flux interpolation grid
    loc_dfr_1d: []f64 = &.{},

    /// Plot points in 1D (equispaced)
    loc_ppts_1d: []f64 = &.{},

    /// Quadrature points in 1D
    loc_qpts_1d: []f64 = &.{},

    /// (i, j) index into `loc_spts_1d` for each solution point
    idx_spts: Matrix(usize) = .empty,

    /// (i, j) index into `loc_dfr_1d` for each flux point.
    ///
    /// Already shifted for the DFR grid: a flux point on the -xi face has
    /// i = 0, one on the +xi face has i = n_spts_1d + 1. ZEFR stores the
    /// unshifted -1 / n_spts_1d and adds one at every use site.
    idx_fpts: Matrix(usize) = .empty,

    /// (i, j) index into `loc_ppts_1d` for each plot point
    idx_ppts: Matrix(usize) = .empty,

    /// (i, j) index into `loc_qpts_1d` for each quadrature point
    idx_qpts: Matrix(usize) = .empty,

    /// Shape-node ordering: `gmsh_to_ijk[n]` is the structured index
    /// `i + n_side*j` of mesh node `n`.
    gmsh_to_ijk: []usize = &.{},

    /// Equispaced shape-node locations in 1D, length `n_side`
    loc_shape_1d: []f64 = &.{},

    const vtable: Element.VTable = .{
        .setLocs = setLocs,
        .setNormals = setNormals,
        .setVandermondeMats = setVandermondeMats,
        .calcShape = calcShape,
        .calcDShape = calcDShape,
        .calcNodalBasis = calcNodalBasis,
        .calcDNodalBasisSpts = calcDNodalBasisSpts,
        .calcDNodalBasisFpts = calcDNodalBasisFpts,
        .calcNodalFaceBasis = calcNodalFaceBasis,
        .calcOrthonormalBasis = calcOrthonormalBasis,
        .getFaceNodes = getFaceNodes,
        .getFaceWeights = getFaceWeights,
        .projectFacePoint = projectFacePoint,
    };

    /// Create a quad element of the given order with `n_nodes` shape nodes per
    /// cell (4 for a linear quad, 9 for a quadratic one, ...).
    ///
    /// Returns an unset element; call `quad.ele.setup()` to build the operators.
    /// The result must not be moved afterwards -- `ele` is recovered by
    /// `@fieldParentPtr`, so the `Quad` has to stay put.
    pub fn init(
        gpa: std.mem.Allocator,
        config: *const Config,
        order: u8,
        n_nodes: usize,
    ) Quad {
        const n_spts_1d: usize = @as(usize, order) + 1;
        const n_qpts_1d: usize = config.test_case.n_qpts_1d;

        return .{ .ele = .{
            .vtable = &vtable,
            .gpa = gpa,
            .config = config,
            .etype = .quad,
            .order = order,
            .n_dims = 2,
            .n_faces = 4,
            .n_nodes = n_nodes,
            .n_spts = n_spts_1d * n_spts_1d,
            .n_fpts = n_spts_1d * 4,
            .n_ppts = n_spts_1d * n_spts_1d,
            .n_qpts = n_qpts_1d * n_qpts_1d,
            .n_spts_1d = n_spts_1d,
            .n_fpts_per_face = n_spts_1d,
        } };
    }

    pub fn deinit(quad: *Quad) void {
        const gpa = quad.ele.gpa;

        gpa.free(quad.loc_spts_1d);
        gpa.free(quad.loc_dfr_1d);
        gpa.free(quad.loc_ppts_1d);
        gpa.free(quad.loc_qpts_1d);

        quad.idx_spts.deinit(gpa);
        quad.idx_fpts.deinit(gpa);
        quad.idx_ppts.deinit(gpa);
        quad.idx_qpts.deinit(gpa);

        gpa.free(quad.gmsh_to_ijk);
        gpa.free(quad.loc_shape_1d);

        quad.ele.deinit();
    }

    // ---- Point locations ----

    fn setLocs(ele: *Element) Element.Error!void {
        const quad: *Quad = @fieldParentPtr("ele", ele);
        const gpa = ele.gpa;

        const n_dims = ele.n_dims;
        const n_spts_1d = ele.n_spts_1d;
        const n_qpts_1d: u32 = @intCast(ele.config.test_case.n_qpts_1d);

        ele.loc_spts = try Matrix(f64).init(gpa, ele.n_spts, n_dims, null);
        ele.loc_fpts = try Matrix(f64).init(gpa, ele.n_fpts, n_dims, null);
        ele.loc_ppts = try Matrix(f64).init(gpa, ele.n_ppts, n_dims, null);
        ele.loc_qpts = try Matrix(f64).init(gpa, ele.n_qpts, n_dims, null);

        quad.idx_spts = try Matrix(usize).init(gpa, ele.n_spts, n_dims, null);
        quad.idx_fpts = try Matrix(usize).init(gpa, ele.n_fpts, n_dims, null);
        quad.idx_ppts = try Matrix(usize).init(gpa, ele.n_ppts, n_dims, null);
        quad.idx_qpts = try Matrix(usize).init(gpa, ele.n_qpts, n_dims, null);

        // --- 1D point sets ---
        // The tabulated sets come back as (N, 1) matrices; their `.data` is the
        // contiguous list, which is what the polynomial routines want.
        quad.loc_spts_1d = (try points.gaussLegendrePts(gpa, @intCast(n_spts_1d))).data;
        const weights_spts_1d = (try points.gaussLegendreWeights(gpa, @intCast(n_spts_1d))).data;
        defer gpa.free(weights_spts_1d);

        // The DFR flux grid: solution points bracketed by the face endpoints
        quad.loc_dfr_1d = try gpa.alloc(f64, n_spts_1d + 2);
        quad.loc_dfr_1d[0] = -1.0;
        @memcpy(quad.loc_dfr_1d[1 .. n_spts_1d + 1], quad.loc_spts_1d);
        quad.loc_dfr_1d[n_spts_1d + 1] = 1.0;

        quad.loc_ppts_1d = (try points.shapePts(gpa, ele.order)).data;
        quad.loc_qpts_1d = (try points.gaussLegendrePts(gpa, n_qpts_1d)).data;
        const weights_qpts_1d = (try points.gaussLegendreWeights(gpa, n_qpts_1d)).data;
        defer gpa.free(weights_qpts_1d);

        // Face flux points use the 1D solution-point rule
        ele.weights_fpts = try gpa.dupe(f64, weights_spts_1d);

        // --- Solution points (x fastest) ---
        ele.weights_spts = try gpa.alloc(f64, ele.n_spts);
        var spt: usize = 0;
        for (0..n_spts_1d) |i| {
            for (0..n_spts_1d) |j| {
                ele.loc_spts.at(spt, 0).* = quad.loc_spts_1d[j];
                ele.loc_spts.at(spt, 1).* = quad.loc_spts_1d[i];
                quad.idx_spts.at(spt, 0).* = j;
                quad.idx_spts.at(spt, 1).* = i;
                ele.weights_spts[spt] = weights_spts_1d[i] * weights_spts_1d[j];
                spt += 1;
            }
        }

        // --- Flux points, face by face, each running counter-clockwise ---
        // idx_fpts holds DFR-grid indices: 0 for a -1 face, n_spts_1d+1 for a
        // +1 face, and j+1 for the tangential position.
        const lo = 0;
        const hi = n_spts_1d + 1;
        var fpt: usize = 0;
        for (0..ele.n_faces) |face| {
            for (0..n_spts_1d) |j| {
                const jr = n_spts_1d - j - 1; // reversed, for the top/left faces
                switch (face) {
                    0 => { // Bottom edge (eta = -1), +xi
                        ele.loc_fpts.at(fpt, 0).* = quad.loc_spts_1d[j];
                        ele.loc_fpts.at(fpt, 1).* = -1.0;
                        quad.idx_fpts.at(fpt, 0).* = j + 1;
                        quad.idx_fpts.at(fpt, 1).* = lo;
                    },
                    1 => { // Right edge (xi = +1), +eta
                        ele.loc_fpts.at(fpt, 0).* = 1.0;
                        ele.loc_fpts.at(fpt, 1).* = quad.loc_spts_1d[j];
                        quad.idx_fpts.at(fpt, 0).* = hi;
                        quad.idx_fpts.at(fpt, 1).* = j + 1;
                    },
                    2 => { // Upper edge (eta = +1), -xi
                        ele.loc_fpts.at(fpt, 0).* = quad.loc_spts_1d[jr];
                        ele.loc_fpts.at(fpt, 1).* = 1.0;
                        quad.idx_fpts.at(fpt, 0).* = jr + 1;
                        quad.idx_fpts.at(fpt, 1).* = hi;
                    },
                    else => { // Left edge (xi = -1), -eta
                        ele.loc_fpts.at(fpt, 0).* = -1.0;
                        ele.loc_fpts.at(fpt, 1).* = quad.loc_spts_1d[jr];
                        quad.idx_fpts.at(fpt, 0).* = lo;
                        quad.idx_fpts.at(fpt, 1).* = jr + 1;
                    },
                }
                fpt += 1;
            }
        }

        // --- Plot points (equispaced) ---
        var ppt: usize = 0;
        for (0..n_spts_1d) |i| {
            for (0..n_spts_1d) |j| {
                ele.loc_ppts.at(ppt, 0).* = quad.loc_ppts_1d[j];
                ele.loc_ppts.at(ppt, 1).* = quad.loc_ppts_1d[i];
                quad.idx_ppts.at(ppt, 0).* = j;
                quad.idx_ppts.at(ppt, 1).* = i;
                ppt += 1;
            }
        }

        // --- Quadrature points ---
        ele.weights_qpts = try gpa.alloc(f64, ele.n_qpts);
        var qpt: usize = 0;
        for (0..n_qpts_1d) |i| {
            for (0..n_qpts_1d) |j| {
                ele.loc_qpts.at(qpt, 0).* = quad.loc_qpts_1d[j];
                ele.loc_qpts.at(qpt, 1).* = quad.loc_qpts_1d[i];
                quad.idx_qpts.at(qpt, 0).* = j;
                quad.idx_qpts.at(qpt, 1).* = i;
                ele.weights_qpts[qpt] = weights_qpts_1d[i] * weights_qpts_1d[j];
                qpt += 1;
            }
        }

        // --- Shape-node layout, for calcShape/calcDShape ---
        const n_side = std.math.sqrt(ele.n_nodes);
        if (n_side * n_side != ele.n_nodes) {
            // Serendipity quads (8 nodes) land here: they have no tensor-product
            // Lagrange basis.
            return error.UnsupportedShapeOrder;
        }
        quad.gmsh_to_ijk = try gmshToStructuredQuad(gpa, ele.n_nodes);
        quad.loc_shape_1d = (try points.shapePts(gpa, @intCast(n_side - 1))).data;
    }

    /// Outward normals in parent space. The reference quad is the bi-unit
    /// square, so every face has unit length scaling.
    fn setNormals(ele: *Element) Element.Error!void {
        const gpa = ele.gpa;

        ele.tnorm = try Matrix(f64).init(gpa, ele.n_fpts, ele.n_dims, null);
        ele.tdA = try gpa.alloc(f64, ele.n_fpts);
        @memset(ele.tdA, 1.0);

        for (0..ele.n_fpts) |fpt| {
            const n: [2]f64 = switch (fpt / ele.n_fpts_per_face) {
                0 => .{ 0.0, -1.0 }, // Bottom edge
                1 => .{ 1.0, 0.0 }, // Right edge
                2 => .{ 0.0, 1.0 }, // Top edge
                else => .{ -1.0, 0.0 }, // Left edge
            };
            ele.tnorm.at(fpt, 0).* = n[0];
            ele.tnorm.at(fpt, 1).* = n[1];
        }
    }

    /// Vandermonde matrix of the 2D Legendre basis at the solution points.
    ///
    /// The inverse is not built: it is only needed by the non-tensor-product
    /// correction operator and by polynomial squeezing, neither of which is
    /// ported.
    fn setVandermondeMats(ele: *Element) Element.Error!void {
        ele.vand = try Matrix(f64).init(ele.gpa, ele.n_spts, ele.n_spts, null);

        for (0..ele.n_spts) |i| {
            const xi = ele.loc_spts.get(i, 0);
            const eta = ele.loc_spts.get(i, 1);
            for (0..ele.n_spts) |j| {
                ele.vand.at(i, j).* = poly.Legendre2D(ele.order, xi, eta, @intCast(j));
            }
        }
    }

    // ---- Geometry (shape) basis ----

    /// Nodal shape basis, written in mesh-file (Gmsh) node order.
    fn calcShape(ele: *const Element, loc: []const f64, shape_val: []f64) Element.Error!void {
        const quad: *const Quad = @fieldParentPtr("ele", ele);
        std.debug.assert(shape_val.len >= ele.n_nodes);

        const n_side = quad.loc_shape_1d.len;
        for (quad.gmsh_to_ijk, 0..) |ijk, node| {
            const i = ijk % n_side;
            const j = ijk / n_side;
            shape_val[node] = poly.Lagrange(quad.loc_shape_1d, loc[0], i) *
                poly.Lagrange(quad.loc_shape_1d, loc[1], j);
        }
    }

    /// Derivatives of the nodal shape basis, (n_nodes, n_dims).
    fn calcDShape(ele: *const Element, loc: []const f64, dshape_val: *Matrix(f64)) Element.Error!void {
        const quad: *const Quad = @fieldParentPtr("ele", ele);
        std.debug.assert(dshape_val.rows >= ele.n_nodes and dshape_val.cols >= ele.n_dims);

        const n_side = quad.loc_shape_1d.len;
        for (quad.gmsh_to_ijk, 0..) |ijk, node| {
            const i = ijk % n_side;
            const j = ijk / n_side;
            dshape_val.at(node, 0).* = poly.dLagrange(quad.loc_shape_1d, loc[0], i) *
                poly.Lagrange(quad.loc_shape_1d, loc[1], j);
            dshape_val.at(node, 1).* = poly.Lagrange(quad.loc_shape_1d, loc[0], i) *
                poly.dLagrange(quad.loc_shape_1d, loc[1], j);
        }
    }

    // ---- Solution basis ----

    /// Tensor-product Lagrange polynomial on the solution points.
    fn calcNodalBasis(ele: *const Element, spt: usize, loc: []const f64) f64 {
        const quad: *const Quad = @fieldParentPtr("ele", ele);
        const i = quad.idx_spts.get(spt, 0);
        const j = quad.idx_spts.get(spt, 1);

        return poly.Lagrange(quad.loc_spts_1d, loc[0], i) *
            poly.Lagrange(quad.loc_spts_1d, loc[1], j);
    }

    /// Derivative of a solution point's basis function, taken on the DFR grid:
    /// the flux is a degree-(P+1) polynomial even though the solution is
    /// degree P.
    fn calcDNodalBasisSpts(ele: *const Element, spt: usize, loc: []const f64, dim: usize) f64 {
        const quad: *const Quad = @fieldParentPtr("ele", ele);
        // Shifted by one: index 0 of the DFR grid is the -1 endpoint.
        const i = quad.idx_spts.get(spt, 0) + 1;
        const j = quad.idx_spts.get(spt, 1) + 1;

        return dfrTensorDeriv(quad.loc_dfr_1d, loc, i, j, dim);
    }

    /// Derivative of a flux point's DFR correction function.
    fn calcDNodalBasisFpts(ele: *const Element, fpt: usize, loc: []const f64, dim: usize) f64 {
        const quad: *const Quad = @fieldParentPtr("ele", ele);
        // idx_fpts already holds DFR-grid indices.
        const i = quad.idx_fpts.get(fpt, 0);
        const j = quad.idx_fpts.get(fpt, 1);

        return dfrTensorDeriv(quad.loc_dfr_1d, loc, i, j, dim);
    }

    /// d/d(loc[dim]) of `L_i(xi) * L_j(eta)` on the DFR grid.
    fn dfrTensorDeriv(grid: []const f64, loc: []const f64, i: usize, j: usize, dim: usize) f64 {
        if (dim == 0) {
            return poly.dLagrange(grid, loc[0], i) * poly.Lagrange(grid, loc[1], j);
        }
        return poly.Lagrange(grid, loc[0], i) * poly.dLagrange(grid, loc[1], j);
    }

    /// A quad face is 1D, so its nodal basis is the 1D solution basis.
    fn calcNodalFaceBasis(ele: *const Element, face: usize, pt: usize, loc: []const f64) f64 {
        const quad: *const Quad = @fieldParentPtr("ele", ele);
        _ = face;
        return poly.Lagrange(quad.loc_spts_1d, loc[0], pt % ele.n_spts_1d);
    }

    fn calcOrthonormalBasis(ele: *const Element, mode: usize, loc: []const f64) f64 {
        return poly.Legendre2D(ele.order, loc[0], loc[1], @intCast(mode));
    }

    // ---- Faces ----

    /// Gauss-Legendre points along a face, exact for degree `p`.
    fn getFaceNodes(
        ele: *const Element,
        gpa: std.mem.Allocator,
        face: usize,
        p: u32,
    ) Element.Error![]f64 {
        _ = ele;
        _ = face; // every face of a quad is the same 1D reference interval
        return (try points.gaussLegendrePts(gpa, p + 1)).data;
    }

    fn getFaceWeights(
        ele: *const Element,
        gpa: std.mem.Allocator,
        face: usize,
        p: u32,
    ) Element.Error![]f64 {
        _ = ele;
        _ = face;
        return (try points.gaussLegendreWeights(gpa, p + 1)).data;
    }

    /// Map a face-local coordinate onto the reference square. The sign flips on
    /// the top and left faces keep every face traversed counter-clockwise,
    /// matching the flux point ordering in `setLocs`.
    fn projectFacePoint(ele: *const Element, face: usize, loc: []const f64, ploc: []f64) void {
        _ = ele;
        switch (face) {
            0 => { // Bottom
                ploc[0] = loc[0];
                ploc[1] = -1.0;
            },
            1 => { // Right
                ploc[0] = 1.0;
                ploc[1] = loc[0];
            },
            2 => { // Top
                ploc[0] = -loc[0];
                ploc[1] = 1.0;
            },
            else => { // Left
                ploc[0] = -1.0;
                ploc[1] = -loc[0];
            },
        }
    }
};

/// Map each Gmsh node of a Lagrange quad to its structured index `i + n_side*j`.
///
/// Gmsh numbers a high-order quad in shells: the four corners of the outermost
/// ring, then its edge nodes, then the next ring in, and a centre node when the
/// side count is odd. Ported from ZEFR's `gmsh_to_structured_quad`.
fn gmshToStructuredQuad(gpa: std.mem.Allocator, n_nodes: usize) Element.Error![]usize {
    const n_side = std.math.sqrt(n_nodes);
    if (n_side * n_side != n_nodes) return error.UnsupportedShapeOrder;

    const map = try gpa.alloc(usize, n_nodes);
    errdefer gpa.free(map);

    const n_levels = n_side / 2;
    var node: usize = 0;
    for (0..n_levels) |i| {
        const hi = (n_side - 1) - i;

        // Corners of this ring, counter-clockwise
        map[node + 0] = i + n_side * i;
        map[node + 1] = hi + n_side * i;
        map[node + 2] = hi + n_side * hi;
        map[node + 3] = i + n_side * hi;
        node += 4;

        // Then its four edges, each running counter-clockwise
        const n_edge = n_side - 2 * (i + 1);
        for (0..n_edge) |j| {
            map[node + j] = (i + 1 + j) + n_side * i;
            map[node + n_edge + j] = hi + n_side * (i + 1 + j);
            map[node + 2 * n_edge + j] = (hi - 1 - j) + n_side * hi;
            map[node + 3 * n_edge + j] = i + n_side * (hi - 1 - j);
        }
        node += 4 * n_edge;
    }

    // Odd side counts leave a single node at the centre
    if (n_side % 2 != 0) {
        map[n_nodes - 1] = n_side / 2 + n_side * (n_side / 2);
    }

    return map;
}

const std = @import("std");

const element = @import("../element.zig");
const Element = element.Element;
const Config = @import("../config.zig").Config;

const Matrix = @import("../util/matrix.zig").Matrix;
const points = @import("../points.zig");
const poly = @import("../math/polynomials.zig");

//! 3D Hexahedral elements
//!
//! The 3D counterpart of `eles/quads.zig`, and structurally the same thing: a
//! triple tensor product of 1D Lagrange bases, with the solution interpolated
//! on `loc_spts_1d` and the flux on `loc_dfr_1d` (the same points bracketed by
//! -1 and +1), which is what makes this DFR rather than plain FR.
//!
//! Ported from ZEFR's `hexas.cpp`. Two conventions are shared with the mesh and
//! must not drift from it:
//!
//!   1. The six faces are ordered bottom (z = -1), top (z = +1), left (x = -1),
//!      right (x = +1), front (y = -1), back (y = +1) -- the order of
//!      `geo.localFaces(.hex)`, which is what makes `lf * n_fpts_per_face` the
//!      right offset into an element's flux points.
//!
//!   2. Within a face, the fast flux-point index runs from the face's first
//!      vertex toward its second, and the slow index from the first toward its
//!      fourth -- again as listed in `geo.localFaces(.hex)`. That is the whole
//!      reason three of the faces traverse their axis backwards.
//!
//! A quad's face is an interval and its two neighbours always meet reversed, so
//! 2D never needed (2) written down. A hex's face is a square that two cells can
//! meet in any of eight relative orientations, so the pairing needs the face's
//! own start and axes, not just a reversal.

pub const Hex = struct {
    ele: Element,

    /// Solution points in 1D (Gauss-Legendre), length `ele.n_spts_1d`
    loc_spts_1d: []f64 = &.{},

    /// `loc_spts_1d` bracketed by -1 and +1; the DFR flux interpolation grid
    loc_dfr_1d: []f64 = &.{},

    /// Plot points in 1D (equispaced)
    loc_ppts_1d: []f64 = &.{},

    /// Quadrature points in 1D
    loc_qpts_1d: []f64 = &.{},

    /// (i, j, k) index into `loc_spts_1d` for each solution point
    idx_spts: Matrix(usize) = .empty,

    /// (i, j, k) index into `loc_dfr_1d` for each flux point.
    ///
    /// Already shifted for the DFR grid: a flux point on a -1 face has 0 in that
    /// direction and one on a +1 face has `n_spts_1d + 1`. ZEFR stores the
    /// unshifted -1 / n_spts_1d and adds one at every use site.
    idx_fpts: Matrix(usize) = .empty,

    /// (i, j, k) index into `loc_ppts_1d` for each plot point
    idx_ppts: Matrix(usize) = .empty,

    /// (i, j, k) index into `loc_qpts_1d` for each quadrature point
    idx_qpts: Matrix(usize) = .empty,

    /// Shape-node ordering: `gmsh_to_ijk[n]` is the structured index
    /// `i + n_side*(j + n_side*k)` of mesh node `n`.
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

    /// Create a hex element of the given order with `n_nodes` shape nodes per
    /// cell (8 for a linear hex, 27 for a quadratic one, ...).
    ///
    /// Returns an unset element; call `hex.ele.setup()` to build the operators.
    /// The result must not be moved afterwards -- `ele` is recovered by
    /// `@fieldParentPtr`, so the `Hex` has to stay put.
    pub fn init(
        gpa: std.mem.Allocator,
        config: *const Config,
        order: u8,
        n_nodes: usize,
    ) Hex {
        const n_spts_1d: usize = @as(usize, order) + 1;
        const n_qpts_1d: usize = config.test_case.n_qpts_1d;

        return .{ .ele = .{
            .vtable = &vtable,
            .gpa = gpa,
            .config = config,
            .etype = .hex,
            .order = order,
            .n_dims = 3,
            .n_faces = 6,
            .n_nodes = n_nodes,
            .n_spts = n_spts_1d * n_spts_1d * n_spts_1d,
            .n_fpts = n_spts_1d * n_spts_1d * 6,
            .n_ppts = n_spts_1d * n_spts_1d * n_spts_1d,
            .n_qpts = n_qpts_1d * n_qpts_1d * n_qpts_1d,
            .n_spts_1d = n_spts_1d,
            .n_fpts_per_face = n_spts_1d * n_spts_1d,
        } };
    }

    pub fn deinit(hex: *Hex) void {
        const gpa = hex.ele.gpa;

        gpa.free(hex.loc_spts_1d);
        gpa.free(hex.loc_dfr_1d);
        gpa.free(hex.loc_ppts_1d);
        gpa.free(hex.loc_qpts_1d);

        hex.idx_spts.deinit(gpa);
        hex.idx_fpts.deinit(gpa);
        hex.idx_ppts.deinit(gpa);
        hex.idx_qpts.deinit(gpa);

        gpa.free(hex.gmsh_to_ijk);
        gpa.free(hex.loc_shape_1d);

        hex.ele.deinit();
    }

    // ---- Face layout ----

    /// How one face sits in the reference cube: which axis it is normal to, and
    /// which axes its two face-local coordinates run along.
    ///
    /// This one table drives the flux point locations, the parent-space normals
    /// and `projectFacePoint`, so those three cannot disagree.
    const FaceLayout = struct {
        /// Cube axis normal to the face
        normal_dim: usize,
        /// Which end of `normal_dim` the face sits at
        normal_hi: bool,
        /// Cube axes the fast and slow face-local indices run along
        tang_dim: [2]usize,
        /// Whether the fast index runs backwards along its axis. Three faces
        /// need it, to start every face's traversal at the first vertex
        /// `geo.localFaces(.hex)` lists for it. The slow index never does.
        fast_rev: bool,
    };

    /// Indexed by local face number, in `geo.localFaces(.hex)` order.
    const face_layout: [6]FaceLayout = .{
        // 0: bottom, z = -1, from vertex 0 toward 1 (+x) then 3 (+y)
        .{ .normal_dim = 2, .normal_hi = false, .tang_dim = .{ 0, 1 }, .fast_rev = false },
        // 1: top, z = +1, from vertex 5 toward 4 (-x) then 6 (+y)
        .{ .normal_dim = 2, .normal_hi = true, .tang_dim = .{ 0, 1 }, .fast_rev = true },
        // 2: left, x = -1, from vertex 0 toward 3 (+y) then 4 (+z)
        .{ .normal_dim = 0, .normal_hi = false, .tang_dim = .{ 1, 2 }, .fast_rev = false },
        // 3: right, x = +1, from vertex 2 toward 1 (-y) then 6 (+z)
        .{ .normal_dim = 0, .normal_hi = true, .tang_dim = .{ 1, 2 }, .fast_rev = true },
        // 4: front, y = -1, from vertex 1 toward 0 (-x) then 5 (+z)
        .{ .normal_dim = 1, .normal_hi = false, .tang_dim = .{ 0, 2 }, .fast_rev = true },
        // 5: back, y = +1, from vertex 3 toward 2 (+x) then 7 (+z)
        .{ .normal_dim = 1, .normal_hi = true, .tang_dim = .{ 0, 2 }, .fast_rev = false },
    };

    // ---- Point locations ----

    fn setLocs(ele: *Element) Element.Error!void {
        const hex: *Hex = @fieldParentPtr("ele", ele);
        const gpa = ele.gpa;

        const n_dims = ele.n_dims;
        const n_spts_1d = ele.n_spts_1d;
        const n_qpts_1d: u32 = @intCast(ele.config.test_case.n_qpts_1d);

        ele.loc_spts = try Matrix(f64).init(gpa, ele.n_spts, n_dims, null);
        ele.loc_fpts = try Matrix(f64).init(gpa, ele.n_fpts, n_dims, null);
        ele.loc_ppts = try Matrix(f64).init(gpa, ele.n_ppts, n_dims, null);
        ele.loc_qpts = try Matrix(f64).init(gpa, ele.n_qpts, n_dims, null);

        hex.idx_spts = try Matrix(usize).init(gpa, ele.n_spts, n_dims, null);
        hex.idx_fpts = try Matrix(usize).init(gpa, ele.n_fpts, n_dims, null);
        hex.idx_ppts = try Matrix(usize).init(gpa, ele.n_ppts, n_dims, null);
        hex.idx_qpts = try Matrix(usize).init(gpa, ele.n_qpts, n_dims, null);

        // --- 1D point sets ---
        // The tabulated sets come back as (N, 1) matrices; their `.data` is the
        // contiguous list, which is what the polynomial routines want.
        hex.loc_spts_1d = (try points.gaussLegendrePts(gpa, @intCast(n_spts_1d))).data;
        const weights_spts_1d = (try points.gaussLegendreWeights(gpa, @intCast(n_spts_1d))).data;
        defer gpa.free(weights_spts_1d);

        // The DFR flux grid: solution points bracketed by the face endpoints
        hex.loc_dfr_1d = try gpa.alloc(f64, n_spts_1d + 2);
        hex.loc_dfr_1d[0] = -1.0;
        @memcpy(hex.loc_dfr_1d[1 .. n_spts_1d + 1], hex.loc_spts_1d);
        hex.loc_dfr_1d[n_spts_1d + 1] = 1.0;

        hex.loc_ppts_1d = (try points.shapePts(gpa, ele.order)).data;
        hex.loc_qpts_1d = (try points.gaussLegendrePts(gpa, n_qpts_1d)).data;
        const weights_qpts_1d = (try points.gaussLegendreWeights(gpa, n_qpts_1d)).data;
        defer gpa.free(weights_qpts_1d);

        // A hex face is a square, so its rule is the 2D tensor product of the
        // 1D solution-point rule -- one weight per flux point on a face.
        ele.weights_fpts = try gpa.alloc(f64, ele.n_fpts_per_face);
        for (0..n_spts_1d) |j| {
            for (0..n_spts_1d) |k| {
                ele.weights_fpts[k + n_spts_1d * j] = weights_spts_1d[j] * weights_spts_1d[k];
            }
        }

        // --- Solution points (x fastest, then y, then z) ---
        ele.weights_spts = try gpa.alloc(f64, ele.n_spts);
        var spt: usize = 0;
        for (0..n_spts_1d) |i| {
            for (0..n_spts_1d) |j| {
                for (0..n_spts_1d) |k| {
                    ele.loc_spts.at(spt, 0).* = hex.loc_spts_1d[k];
                    ele.loc_spts.at(spt, 1).* = hex.loc_spts_1d[j];
                    ele.loc_spts.at(spt, 2).* = hex.loc_spts_1d[i];
                    hex.idx_spts.at(spt, 0).* = k;
                    hex.idx_spts.at(spt, 1).* = j;
                    hex.idx_spts.at(spt, 2).* = i;
                    ele.weights_spts[spt] =
                        weights_spts_1d[i] * weights_spts_1d[j] * weights_spts_1d[k];
                    spt += 1;
                }
            }
        }

        // --- Flux points, face by face ---
        // idx_fpts holds DFR-grid indices: 0 on a -1 face, n_spts_1d+1 on a +1
        // face, and (tangential index + 1) along the face's own two axes.
        const lo = 0;
        const hi = n_spts_1d + 1;
        var fpt: usize = 0;
        for (0..ele.n_faces) |face| {
            const fl = face_layout[face];
            for (0..n_spts_1d) |j| {
                for (0..n_spts_1d) |k| {
                    const kk = if (fl.fast_rev) n_spts_1d - 1 - k else k;

                    ele.loc_fpts.at(fpt, fl.normal_dim).* = if (fl.normal_hi) 1.0 else -1.0;
                    ele.loc_fpts.at(fpt, fl.tang_dim[0]).* = hex.loc_spts_1d[kk];
                    ele.loc_fpts.at(fpt, fl.tang_dim[1]).* = hex.loc_spts_1d[j];

                    hex.idx_fpts.at(fpt, fl.normal_dim).* = if (fl.normal_hi) hi else lo;
                    hex.idx_fpts.at(fpt, fl.tang_dim[0]).* = kk + 1;
                    hex.idx_fpts.at(fpt, fl.tang_dim[1]).* = j + 1;

                    fpt += 1;
                }
            }
        }

        // --- Plot points (equispaced) ---
        var ppt: usize = 0;
        for (0..n_spts_1d) |i| {
            for (0..n_spts_1d) |j| {
                for (0..n_spts_1d) |k| {
                    ele.loc_ppts.at(ppt, 0).* = hex.loc_ppts_1d[k];
                    ele.loc_ppts.at(ppt, 1).* = hex.loc_ppts_1d[j];
                    ele.loc_ppts.at(ppt, 2).* = hex.loc_ppts_1d[i];
                    hex.idx_ppts.at(ppt, 0).* = k;
                    hex.idx_ppts.at(ppt, 1).* = j;
                    hex.idx_ppts.at(ppt, 2).* = i;
                    ppt += 1;
                }
            }
        }

        // --- Quadrature points ---
        ele.weights_qpts = try gpa.alloc(f64, ele.n_qpts);
        var qpt: usize = 0;
        for (0..n_qpts_1d) |i| {
            for (0..n_qpts_1d) |j| {
                for (0..n_qpts_1d) |k| {
                    ele.loc_qpts.at(qpt, 0).* = hex.loc_qpts_1d[k];
                    ele.loc_qpts.at(qpt, 1).* = hex.loc_qpts_1d[j];
                    ele.loc_qpts.at(qpt, 2).* = hex.loc_qpts_1d[i];
                    hex.idx_qpts.at(qpt, 0).* = k;
                    hex.idx_qpts.at(qpt, 1).* = j;
                    hex.idx_qpts.at(qpt, 2).* = i;
                    ele.weights_qpts[qpt] =
                        weights_qpts_1d[i] * weights_qpts_1d[j] * weights_qpts_1d[k];
                    qpt += 1;
                }
            }
        }

        // --- Shape-node layout, for calcShape/calcDShape ---
        // Serendipity hexes (20 nodes) fail the cube test: they have no
        // tensor-product Lagrange basis. Below 8 nodes there is no hex at all,
        // and `n_side - 1` would wrap.
        const n_side = intCbrt(ele.n_nodes) orelse return error.UnsupportedShapeOrder;
        if (n_side < 2) return error.UnsupportedShapeOrder;
        hex.gmsh_to_ijk = try gmshToStructuredHex(gpa, ele.n_nodes);
        hex.loc_shape_1d = (try points.shapePts(gpa, @intCast(n_side - 1))).data;
    }

    /// Outward normals in parent space. The reference hex is the bi-unit cube,
    /// so every face has unit area scaling.
    fn setNormals(ele: *Element) Element.Error!void {
        const gpa = ele.gpa;

        // Matrix.init zeroes, so only the one non-zero component is written.
        ele.tnorm = try Matrix(f64).init(gpa, ele.n_fpts, ele.n_dims, null);
        ele.tdA = try gpa.alloc(f64, ele.n_fpts);
        @memset(ele.tdA, 1.0);

        for (0..ele.n_fpts) |fpt| {
            const fl = face_layout[fpt / ele.n_fpts_per_face];
            ele.tnorm.at(fpt, fl.normal_dim).* = if (fl.normal_hi) 1.0 else -1.0;
        }
    }

    /// Vandermonde matrix of the 3D Legendre basis at the solution points.
    ///
    /// The inverse is not built: it is only needed by the non-tensor-product
    /// correction operator and by polynomial squeezing, neither of which is
    /// ported.
    fn setVandermondeMats(ele: *Element) Element.Error!void {
        ele.vand = try Matrix(f64).init(ele.gpa, ele.n_spts, ele.n_spts, null);

        for (0..ele.n_spts) |i| {
            const xi = ele.loc_spts.get(i, 0);
            const eta = ele.loc_spts.get(i, 1);
            const mu = ele.loc_spts.get(i, 2);
            for (0..ele.n_spts) |j| {
                ele.vand.at(i, j).* = poly.Legendre3D(ele.order, xi, eta, mu, @intCast(j));
            }
        }
    }

    // ---- Geometry (shape) basis ----

    /// Nodal shape basis, written in mesh-file (Gmsh) node order.
    fn calcShape(ele: *const Element, loc: []const f64, shape_val: []f64) Element.Error!void {
        const hex: *const Hex = @fieldParentPtr("ele", ele);
        std.debug.assert(shape_val.len >= ele.n_nodes);

        const n_side = hex.loc_shape_1d.len;
        for (hex.gmsh_to_ijk, 0..) |ijk, node| {
            const i = ijk % n_side;
            const j = (ijk / n_side) % n_side;
            const k = ijk / (n_side * n_side);
            shape_val[node] = poly.Lagrange(hex.loc_shape_1d, loc[0], i) *
                poly.Lagrange(hex.loc_shape_1d, loc[1], j) *
                poly.Lagrange(hex.loc_shape_1d, loc[2], k);
        }
    }

    /// Derivatives of the nodal shape basis, (n_nodes, n_dims).
    fn calcDShape(ele: *const Element, loc: []const f64, dshape_val: *Matrix(f64)) Element.Error!void {
        const hex: *const Hex = @fieldParentPtr("ele", ele);
        std.debug.assert(dshape_val.rows >= ele.n_nodes and dshape_val.cols >= ele.n_dims);

        const n_side = hex.loc_shape_1d.len;
        for (hex.gmsh_to_ijk, 0..) |ijk, node| {
            const idx: [3]usize = .{
                ijk % n_side,
                (ijk / n_side) % n_side,
                ijk / (n_side * n_side),
            };
            for (0..3) |dim| {
                var val: f64 = 1.0;
                for (0..3) |d| {
                    val *= if (d == dim)
                        poly.dLagrange(hex.loc_shape_1d, loc[d], idx[d])
                    else
                        poly.Lagrange(hex.loc_shape_1d, loc[d], idx[d]);
                }
                dshape_val.at(node, dim).* = val;
            }
        }
    }

    // ---- Solution basis ----

    /// Tensor-product Lagrange polynomial on the solution points.
    fn calcNodalBasis(ele: *const Element, spt: usize, loc: []const f64) f64 {
        const hex: *const Hex = @fieldParentPtr("ele", ele);

        var val: f64 = 1.0;
        for (0..3) |d| {
            val *= poly.Lagrange(hex.loc_spts_1d, loc[d], hex.idx_spts.get(spt, d));
        }
        return val;
    }

    /// Derivative of a solution point's basis function, taken on the DFR grid:
    /// the flux is a degree-(P+1) polynomial even though the solution is
    /// degree P.
    fn calcDNodalBasisSpts(ele: *const Element, spt: usize, loc: []const f64, dim: usize) f64 {
        const hex: *const Hex = @fieldParentPtr("ele", ele);
        // Shifted by one: index 0 of the DFR grid is the -1 endpoint.
        const idx: [3]usize = .{
            hex.idx_spts.get(spt, 0) + 1,
            hex.idx_spts.get(spt, 1) + 1,
            hex.idx_spts.get(spt, 2) + 1,
        };
        return dfrTensorDeriv(hex.loc_dfr_1d, loc, idx, dim);
    }

    /// Derivative of a flux point's DFR correction function.
    fn calcDNodalBasisFpts(ele: *const Element, fpt: usize, loc: []const f64, dim: usize) f64 {
        const hex: *const Hex = @fieldParentPtr("ele", ele);
        // idx_fpts already holds DFR-grid indices.
        const idx: [3]usize = .{
            hex.idx_fpts.get(fpt, 0),
            hex.idx_fpts.get(fpt, 1),
            hex.idx_fpts.get(fpt, 2),
        };
        return dfrTensorDeriv(hex.loc_dfr_1d, loc, idx, dim);
    }

    /// d/d(loc[dim]) of `L_i(xi) * L_j(eta) * L_k(mu)` on the DFR grid.
    fn dfrTensorDeriv(grid: []const f64, loc: []const f64, idx: [3]usize, dim: usize) f64 {
        var val: f64 = 1.0;
        for (0..3) |d| {
            val *= if (d == dim)
                poly.dLagrange(grid, loc[d], idx[d])
            else
                poly.Lagrange(grid, loc[d], idx[d]);
        }
        return val;
    }

    /// A hex face is 2D, so its nodal basis is the 2D solution basis. `pt`
    /// indexes the face's flux points in the layout `setLocs` builds: fast
    /// index first.
    fn calcNodalFaceBasis(ele: *const Element, face: usize, pt: usize, loc: []const f64) f64 {
        const hex: *const Hex = @fieldParentPtr("ele", ele);
        _ = face; // every face of a hex is the same reference square
        return poly.Lagrange(hex.loc_spts_1d, loc[0], pt % ele.n_spts_1d) *
            poly.Lagrange(hex.loc_spts_1d, loc[1], pt / ele.n_spts_1d);
    }

    fn calcOrthonormalBasis(ele: *const Element, mode: usize, loc: []const f64) f64 {
        return poly.Legendre3D(ele.order, loc[0], loc[1], loc[2], @intCast(mode));
    }

    // ---- Faces ----

    /// Gauss-Legendre points on a face, exact for degree `p`. Flat
    /// `((p+1)^2, 2)` row-major, matching a face-local `(loc[0], loc[1])`.
    fn getFaceNodes(
        ele: *const Element,
        gpa: std.mem.Allocator,
        face: usize,
        p: u32,
    ) Element.Error![]f64 {
        _ = ele;
        _ = face; // every face of a hex is the same reference square

        const pts_1d = (try points.gaussLegendrePts(gpa, p + 1)).data;
        defer gpa.free(pts_1d);

        const n = pts_1d.len;
        const out = try gpa.alloc(f64, n * n * 2);
        var pt: usize = 0;
        for (0..n) |i| {
            for (0..n) |j| {
                out[2 * pt] = pts_1d[j];
                out[2 * pt + 1] = pts_1d[i];
                pt += 1;
            }
        }
        return out;
    }

    /// Weights matching `getFaceNodes`, one per point.
    fn getFaceWeights(
        ele: *const Element,
        gpa: std.mem.Allocator,
        face: usize,
        p: u32,
    ) Element.Error![]f64 {
        _ = ele;
        _ = face;

        const wts_1d = (try points.gaussLegendreWeights(gpa, p + 1)).data;
        defer gpa.free(wts_1d);

        const n = wts_1d.len;
        const out = try gpa.alloc(f64, n * n);
        for (0..n) |i| {
            for (0..n) |j| out[j + n * i] = wts_1d[i] * wts_1d[j];
        }
        return out;
    }

    /// Map a face-local coordinate onto the reference cube. The sign flip on
    /// the three faces with `fast_rev` is the continuous form of the reversed
    /// flux point traversal in `setLocs`, so a face point and the flux point
    /// with the same face-local index land in the same place.
    fn projectFacePoint(ele: *const Element, face: usize, loc: []const f64, ploc: []f64) void {
        _ = ele;
        const fl = face_layout[face];
        ploc[fl.normal_dim] = if (fl.normal_hi) 1.0 else -1.0;
        ploc[fl.tang_dim[0]] = if (fl.fast_rev) -loc[0] else loc[0];
        ploc[fl.tang_dim[1]] = loc[1];
    }
};

/// Exact integer cube root, or null when `n` is not a perfect cube.
fn intCbrt(n: usize) ?usize {
    var s: usize = 0;
    while (s * s * s < n) s += 1;
    return if (s * s * s == n) s else null;
}

/// Map each Gmsh node of a Lagrange hex to its structured index
/// `i + n_side*(j + n_side*k)`.
///
/// Gmsh numbers a high-order hex in nested shells: the eight corners of the
/// outermost shell, then its twelve edges, then its six faces (each ordered as
/// a Gmsh quad in its own right), then the next shell in, and a centre node
/// when the side count is odd. Ported from ZEFR's `gmsh_to_structured_hex`.
///
/// The face block runs bottom, front, left, right, back, top -- Gmsh's face
/// order, which is *not* the solver's face order in `Hex.face_layout`. The two
/// are unrelated: this one describes where mesh nodes sit, that one where flux
/// points sit.
fn gmshToStructuredHex(gpa: std.mem.Allocator, n_nodes: usize) Element.Error![]usize {
    const n_side = intCbrt(n_nodes) orelse return error.UnsupportedShapeOrder;

    const map = try gpa.alloc(usize, n_nodes);
    errdefer gpa.free(map);

    const idx = struct {
        fn at(n: usize, i: usize, j: usize, k: usize) usize {
            return i + n * (j + n * k);
        }
    }.at;

    const n_levels = n_side / 2;
    var node: usize = 0;

    for (0..n_levels) |i| {
        const hi = (n_side - 1) - i;

        // --- The shell's eight corners, in Gmsh's hex vertex order ---
        map[node + 0] = idx(n_side, i, i, i);
        map[node + 1] = idx(n_side, hi, i, i);
        map[node + 2] = idx(n_side, hi, hi, i);
        map[node + 3] = idx(n_side, i, hi, i);
        map[node + 4] = idx(n_side, i, i, hi);
        map[node + 5] = idx(n_side, hi, i, hi);
        map[node + 6] = idx(n_side, hi, hi, hi);
        map[node + 7] = idx(n_side, i, hi, hi);
        node += 8;

        // --- Its twelve edges, in Gmsh's edge order ---
        // (0,1) (0,3) (0,4) (1,2) (1,5) (2,3) (2,6) (3,7) (4,5) (4,7) (5,6) (6,7)
        const n_edge = n_side - 2 * (i + 1);
        for (0..n_edge) |j| {
            const lo_j = i + 1 + j; // forward along the edge
            const hi_j = hi - 1 - j; // backward along it

            // Around the bottom (z = lo)
            map[node + 0 * n_edge + j] = idx(n_side, lo_j, i, i);
            map[node + 1 * n_edge + j] = idx(n_side, i, lo_j, i);
            map[node + 3 * n_edge + j] = idx(n_side, hi, lo_j, i);
            map[node + 5 * n_edge + j] = idx(n_side, hi_j, hi, i);

            // The four vertical edges
            map[node + 2 * n_edge + j] = idx(n_side, i, i, lo_j);
            map[node + 4 * n_edge + j] = idx(n_side, hi, i, lo_j);
            map[node + 6 * n_edge + j] = idx(n_side, hi, hi, lo_j);
            map[node + 7 * n_edge + j] = idx(n_side, i, hi, lo_j);

            // Around the top (z = hi)
            map[node + 8 * n_edge + j] = idx(n_side, lo_j, i, hi);
            map[node + 9 * n_edge + j] = idx(n_side, i, lo_j, hi);
            map[node + 10 * n_edge + j] = idx(n_side, hi, lo_j, hi);
            map[node + 11 * n_edge + j] = idx(n_side, hi_j, hi, hi);
        }
        node += 12 * n_edge;

        // --- Its six faces, each numbered like a Gmsh quad of side n_edge ---
        //
        // Only where a face's in-plane pair lands in the cube differs between
        // the six, so that is all `FacePlace` carries and `quadShells` does the
        // rest. Gmsh's face order here is bottom, front, left, right, back, top.
        const face_lo = i + 1;
        const face_hi = i + n_edge;
        const faces: [6]FacePlace = .{
            .{ .n_side = n_side, .fixed_dim = 2, .fixed = i, .dim = .{ 1, 0 } },
            .{ .n_side = n_side, .fixed_dim = 1, .fixed = i, .dim = .{ 0, 2 } },
            .{ .n_side = n_side, .fixed_dim = 0, .fixed = i, .dim = .{ 2, 1 } },
            .{ .n_side = n_side, .fixed_dim = 0, .fixed = hi, .dim = .{ 1, 2 } },
            .{
                .n_side = n_side,
                .fixed_dim = 1,
                .fixed = hi,
                .dim = .{ 0, 2 },
                .mirror = face_lo + face_hi,
            },
            .{ .n_side = n_side, .fixed_dim = 2, .fixed = hi, .dim = .{ 0, 1 } },
        };

        for (faces) |place| {
            node += quadShells(map[node..], n_edge, face_lo, place);
        }
    }

    // Odd side counts leave a single node at the very centre
    if (n_side % 2 != 0) {
        const c = n_side / 2;
        map[n_nodes - 1] = idx(n_side, c, c, c);
    }

    return map;
}

/// Where one hex face's in-plane index pair `(a, b)` sits in the cube.
///
/// Gmsh does not orient the six faces alike -- three of them come out with
/// their in-plane axes transposed or one axis reversed relative to the other
/// three -- so the mapping is data rather than a formula.
const FacePlace = struct {
    n_side: usize,
    /// Cube axis the face is normal to, and its fixed index along it
    fixed_dim: usize,
    fixed: usize,
    /// Cube axis each of `a` and `b` runs along
    dim: [2]usize,
    /// `lo + hi` when `a` runs backwards along its axis, so that the reflection
    /// is `mirror - a`; zero when it runs forwards.
    mirror: usize = 0,

    fn at(p: FacePlace, a: usize, b: usize) usize {
        var ijk: [3]usize = undefined;
        ijk[p.fixed_dim] = p.fixed;
        ijk[p.dim[0]] = if (p.mirror != 0) p.mirror - a else a;
        ijk[p.dim[1]] = b;
        return ijk[0] + p.n_side * (ijk[1] + p.n_side * ijk[2]);
    }
};

/// Number the interior of one hex face the way Gmsh numbers a quad: rings of
/// four corners then four edges, working inward, then a centre node if the ring
/// count is odd.
///
/// `n` is the face's side count, `off` the offset of its first point along both
/// in-plane directions, and `place` says where an in-plane pair lands. Returns
/// how many nodes were written.
fn quadShells(map: []usize, n: usize, off: usize, place: FacePlace) usize {
    var node: usize = 0;

    for (0..n / 2) |r| {
        const lo = off + r;
        const hi = off + (n - 1) - r;

        // Corners of this ring, counter-clockwise in (a, b)
        map[node + 0] = place.at(lo, lo);
        map[node + 1] = place.at(hi, lo);
        map[node + 2] = place.at(hi, hi);
        map[node + 3] = place.at(lo, hi);
        node += 4;

        // Then its four edges, each running counter-clockwise
        const n_edge = n - 2 * (r + 1);
        for (0..n_edge) |k| {
            map[node + 0 * n_edge + k] = place.at(lo + 1 + k, lo);
            map[node + 1 * n_edge + k] = place.at(hi, lo + 1 + k);
            map[node + 2 * n_edge + k] = place.at(hi - 1 - k, hi);
            map[node + 3 * n_edge + k] = place.at(lo, hi - 1 - k);
        }
        node += 4 * n_edge;
    }

    // Odd ring counts leave a single node at the face's centre
    if (n % 2 != 0) {
        const c = off + n / 2;
        map[node] = place.at(c, c);
        node += 1;
    }

    return node;
}

const std = @import("std");

const element = @import("../element.zig");
const Element = element.Element;
const Config = @import("../config.zig").Config;

const Matrix = @import("../util/matrix.zig").Matrix;
const points = @import("../points.zig");
const poly = @import("../math/polynomials.zig");

//! 2D Quadrilateral elements
const std = @import("std");

const Matrix = @import("../matrix.zig").Matrix;
const Array3 = @import("../array3.zig").Array3;
const Array4 = @import("../array4.zig").Array4;
const points = @import("../points.zig");
const poly = @import("../math/polynomials.zig");

const Element = @import("../element.zig").Element;

const Quad = @This();

ele: *Element,

n_spts_1d: usize = 0,
loc_spts_1d: std.ArrayList(f64) = .empty,
idx_spts: Matrix(usize) = undefined,
idx_fpts: Matrix(usize) = undefined,
loc_dfr_1d: std.ArrayList(f64) = .empty,

pub fn init(order: u8) Quad {
    const ele = Element{
        .n_spts = (order + 1) * (order + 1),
        .n_fpts = (order + 1) * 4,
        .n_dims = 2,
        .n_faces = 4,
        .vtable = .{
            .setup = setup,
            .set_vandermonde_mats = setVandermondeMats,
            .eval_shape = shape,
            .eval_d_shape = dShape,
            .eval_nodal_basis = nodalBasis,
            .eval_d_nodal_basis_spts = dNodalBasisSpts,
            .eval_d_nodal_basis_fpts = dNodalBasisFpts,
            .get_face_nodes = getFaceNodes,
            .get_face_weights = getFaceWeights,
            .project_face_point = projectFacePoint,
        },
    };

    return .{
        .ele = ele,
        .n_spts_1d = (order + 1),
    };
}

fn setup(ele: *Element) void {
    const quad: *Quad = @fieldParentPtr("ele", ele);
    const gpa = ele.gpa;

    const n_spts = ele.n_spts;
    const n_fpts = ele.n_fpts;
    const n_ppts = ele.n_ppts;
    const n_qpts = ele.n_qpts;
    const order = ele.order;
    const n_spts_1d = quad.n_spts_1d;
    const n_dims = 2;

    // Allocate memory for point location structures
    ele.loc_spts = .init(gpa, n_spts, n_dims) catch @panic("OOM");
    ele.loc_fpts = .init(gpa, n_fpts, n_dims) catch @panic("OOM");
    ele.loc_ppts = .init(gpa, n_ppts, n_dims) catch @panic("OOM");
    ele.loc_qpts = .init(gpa, n_qpts, n_dims) catch @panic("OOM");
    quad.idx_spts = .init(n_spts, n_dims) catch @panic("OOM");
    quad.idx_fpts = .init(n_fpts, n_dims) catch @panic("OOM");
    quad.idx_ppts = .init(n_ppts, n_dims) catch @panic("OOM");
    quad.idx_qpts = .init(n_qpts, n_dims) catch @panic("OOM");

    // Get positions of points in 1D
    quad.loc_spts_1d = points.gaussLegendrePts(gpa, order + 1);
    const loc_spts_1d = quad.loc_spts_1d;

    // // NOTE: Currently assuming solution point locations always at Legendre.
    // // Will need extrapolation operation in 1D otherwise
    // const weights_spts_1D = gaussLegendreWeights(n_spts_1d);
    // weights_fpts.assign({n_spts_1d});
    // for (unsigned int fpt = 0; fpt < n_spts_1d; fpt++)
    //   weights_fpts(fpt) = weights_spts_1D[fpt];

    // loc_DFR_1D = loc_spts_1D;
    // loc_DFR_1D.insert(loc_DFR_1D.begin(), -1.0);
    // loc_DFR_1D.insert(loc_DFR_1D.end(), 1.0);

    // Setup solution point locations and quadrature weights
    // weights_spts.assign({nSpts});
    var spt: usize = 0;
    for (0..n_spts_1d) |i| {
        for (0..n_spts_1d) |j| {
            ele.loc_spts.at(spt, 0).* = loc_spts_1d[j];
            ele.loc_spts.at(spt, 1).* = loc_spts_1d[i];
            quad.idx_spts.at(spt, 0).* = j;
            quad.idx_spts.at(spt, 1).* = i;
            // weights_spts(spt) = weights_spts_1D[i] * weights_spts_1D[j];
            spt += 1;
        }
    }

    // Setup flux point locations
    var fpt: usize = 0;
    for (0..ele.n_faces) |i| {
        for (0..ele.n_spts) |j| {
            switch (i) {
                0 => {
                    // Bottom edge
                    ele.loc_fpts.at(fpt, 0).* = loc_spts_1d[j];
                    ele.loc_fpts.at(fpt, 1).* = -1.0;
                    quad.idx_fpts.at(fpt, 0).* = j;
                    quad.idx_fpts.at(fpt, 1).* = -1;
                },
                1 => {
                    // Right edge
                    ele.loc_fpts.at(fpt, 0).* = 1.0;
                    ele.loc_fpts.at(fpt, 1).* = loc_spts_1d[j];
                    quad.idx_fpts.at(fpt, 0).* = n_spts_1d;
                    quad.idx_fpts.at(fpt, 1).* = j;
                },
                2 => {
                    // * Upper edge
                    ele.loc_fpts.at(fpt, 0).* = loc_spts_1d[n_spts_1d - j - 1];
                    ele.loc_fpts.at(fpt, 1).* = 1.0;
                    quad.idx_fpts.at(fpt, 0).* = n_spts_1d - j - 1;
                    quad.idx_fpts.at(fpt, 1).* = n_spts_1d;
                },
                3 => {
                    // Left edge
                    ele.loc_fpts.at(fpt, 0).* = -1.0;
                    ele.loc_fpts.at(fpt, 1).* = loc_spts_1d[n_spts_1d - j - 1];
                    quad.idx_fpts.at(fpt, 0).* = -1;
                    quad.idx_fpts.at(fpt, 1).* = n_spts_1d - j - 1;
                },
            }

            fpt += 1;
        }
    }
}

pub fn setVandermondeMats(ele: *Element) void {
    _ = ele; // autofix
}

/// Evaluate the nodal shape basis at a reference location
pub fn shape(ele: *Element, loc: []const f64, shape_vals: Matrix(f64)) void {
    _ = ele; // autofix
    _ = loc; // autofix
    _ = shape_vals; // autofix
}
/// Evaluate the derivative of the nodal shape basis at a reference location
pub fn dShape(ele: *Element, loc: []const f64, dshape_val: Matrix(f64)) void {
    _ = ele; // autofix
    _ = loc; // autofix
    _ = dshape_val; // autofix
}

/// Evaluate the nodal basis
/// Tensor-produce Lagrange polynomial defined at the Gauss-Legendre points
pub fn nodalBasis(ele: *const Element, spt: u32, loc: []const f64) f64 {
    // Get indices for Lagrange polynomial evaluation
    const quad: *Quad = @fieldParentPtr("ele", ele);
    const i = quad.idx_spts.get(spt, 0);
    const j = quad.idx_spts.get(spt, 1);

    return poly.Lagrange(quad.loc_spts_1d, i, loc[0]) * poly.Lagrange(quad.loc_spts_1d, j, loc[1]);
}

pub fn dNodalBasisSpts(ele: *const Element, spt: u32, loc: []const f64, dim: u32) f64 {
    // Get indices for Lagrange polynomial evaluation
    // (shifted due to inclusion of boundary points for DFR)
    const quad: *Quad = @fieldParentPtr("ele", ele);
    const i = quad.idx_spts.get(spt, 0) + 1;
    const j = quad.idx_spts.get(spt, 1) + 1;

    if (dim == 0)
        return poly.dLagrange(quad.loc_dfr_1d, i, loc[0]) * poly.Lagrange(quad.loc_dfr_1d, j, loc[1]);
    return poly.Lagrange(quad.loc_dfr_1d, i, loc[0]) * poly.dLagrange(quad.loc_dfr_1d, j, loc[1]);
}

pub fn dNodalBasisFpts(ele: *const Element, fpt: u32, loc: []const f64, dim: u32) f64 {
    // Get indices for Lagrange polynomial evaluation
    // (shifted due to inclusion of boundary points for DFR)
    const quad: *Quad = @fieldParentPtr("ele", ele);
    const i = quad.idx_fpts.get(fpt, 0) + 1;
    const j = quad.idx_fpts.get(fpt, 1) + 1;

    if (dim == 0)
        return poly.dLagrange(quad.loc_dfr_1d, i, loc[0]) * poly.Lagrange(quad.loc_dfr_1d, j, loc[1]);
    return poly.Lagrange(quad.loc_dfr_1d, i, loc[0]) * poly.dLagrange(quad.loc_dfr_1d, j, loc[1]);
}

pub fn getFaceNodes(ele: *const Element, face: u32, P: u32) Matrix(f64) {
    _ = ele; // autofix
    _ = face; // autofix
    _ = P; // autofix
}
pub fn getFaceWeights(ele: *const Element, face: u32, P: u32) Matrix(f64) {
    _ = ele; // autofix
    _ = face; // autofix
    _ = P; // autofix
}

pub fn projectFacePoint(ele: *const Element, face: u32, loc: []const f64, ploc: []f64) void {
    _ = ele; // autofix
    _ = face; // autofix
    _ = loc; // autofix
    _ = ploc; // autofix
}

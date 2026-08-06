//! Element base type: the geometry-independent half of the Direct Flux
//! Reconstruction (DFR) scheme.
//!
//! An element *implementation* (see `eles/quads.zig`) owns an `Element` and
//! supplies a `VTable` describing its reference-space point layout and nodal
//! basis. `Element.setup` then builds the operators the solver runs:
//!
//!   oppE         spts -> fpts extrapolation               (n_fpts, n_spts)
//!   oppD         gradient at spts, per dimension          (n_dims, n_spts, n_spts)
//!   oppD_fpts    gradient at spts from fpt values         (n_dims, n_spts, n_fpts)
//!   oppDiv       divergence at spts                       (n_spts, n_dims, n_spts)
//!   oppDiv_fpts  divergence correction from normal fluxes (n_spts, n_fpts)
//!   oppE_ppts    spts -> plot points                      (n_ppts, n_spts)
//!   oppE_qpts    spts -> quadrature points                (n_qpts, n_spts)
//!
//! Ported from ZEFR (`elements.cpp`, `quads.cpp`), restricted to the
//! tensor-product (quad/hex) DFR path. Overset, moving grids, filtering,
//! p-multigrid and implicit Jacobians are all left out.

pub const Element = struct {
    /// Errors raised while setting up an element.
    pub const Error = error{
        OutOfMemory,
        /// A tabulated point set does not cover the requested order
        UnsupportedOrder,
        /// The element's shape-node count is not a supported Lagrange layout
        UnsupportedShapeOrder,
    };

    /// Per-element-type behaviour. Implementations recover themselves with
    /// `@fieldParentPtr("ele", ele)`.
    pub const VTable = struct {
        /// Fill the reference-space point locations, indices and weights
        setLocs: *const fn (*Element) Error!void,

        /// Fill the parent-space face normals (`tnorm`) and face scalings (`tdA`)
        setNormals: *const fn (*Element) Error!void,

        /// Fill the Vandermonde matrix of the orthonormal modal basis
        setVandermondeMats: *const fn (*Element) Error!void,

        /// Evaluate the nodal shape (geometry) basis at a reference location.
        /// `shape_val` has one entry per shape node, in mesh-file node order.
        calcShape: *const fn (*const Element, loc: []const f64, shape_val: []f64) Error!void,

        /// Derivatives of the nodal shape basis; `dshape_val` is (n_nodes, n_dims)
        calcDShape: *const fn (*const Element, loc: []const f64, dshape_val: *Matrix(f64)) Error!void,

        /// Evaluate solution-point nodal basis function `spt` at `loc`
        calcNodalBasis: *const fn (*const Element, spt: usize, loc: []const f64) f64,

        /// d/d(loc[dim]) of the basis function for solution point `spt`
        calcDNodalBasisSpts: *const fn (*const Element, spt: usize, loc: []const f64, dim: usize) f64,

        /// d/d(loc[dim]) of the DFR correction basis function for flux point `fpt`
        calcDNodalBasisFpts: *const fn (*const Element, fpt: usize, loc: []const f64, dim: usize) f64,

        /// Face-local nodal basis, for the non-tensor-product correction operator
        calcNodalFaceBasis: *const fn (*const Element, face: usize, pt: usize, loc: []const f64) f64,

        /// Orthonormal (modal) basis function `mode` at `loc`
        calcOrthonormalBasis: *const fn (*const Element, mode: usize, loc: []const f64) f64,

        /// Quadrature points on a face for a rule of order `p`. Caller owns.
        getFaceNodes: *const fn (*const Element, std.mem.Allocator, face: usize, p: u32) Error![]f64,

        /// Quadrature weights matching `getFaceNodes`. Caller owns.
        getFaceWeights: *const fn (*const Element, std.mem.Allocator, face: usize, p: u32) Error![]f64,

        /// Lift a face-local reference location onto the element's reference face
        projectFacePoint: *const fn (*const Element, face: usize, loc: []const f64, ploc: []f64) void,
    };

    vtable: *const VTable,

    gpa: std.mem.Allocator,
    config: *const Config,

    etype: CellType,
    order: u8 = 0,

    n_dims: u8 = 0,
    n_faces: u8 = 0,

    /// Shape (mesh) nodes per element, i.e. `geo.c2nv` for this cell
    n_nodes: usize = 0,

    n_spts: usize = 0,
    n_fpts: usize = 0,
    n_ppts: usize = 0,
    n_qpts: usize = 0,

    /// Solution points along one reference direction. Tensor-product only.
    n_spts_1d: usize = 0,
    /// Flux points on a single face
    n_fpts_per_face: usize = 0,

    // ---- Point Locations (reference space) ----

    /// Solution Points, (n_spts, n_dims)
    loc_spts: Matrix(f64) = .empty,

    /// Flux Points, (n_fpts, n_dims)
    loc_fpts: Matrix(f64) = .empty,

    /// Plot Points, equispaced, (n_ppts, n_dims)
    loc_ppts: Matrix(f64) = .empty,

    /// Quadrature Points, (n_qpts, n_dims)
    loc_qpts: Matrix(f64) = .empty,

    // ---- Quadrature Weights ----

    /// Weight per solution point; the spts double as a quadrature rule
    weights_spts: []f64 = &.{},

    /// Quadrature weight per flux point on one face, length `n_fpts_per_face`.
    /// A quad's face is an interval, so this is the 1D rule; a hex's is a
    /// square, so it is that rule's 2D tensor product.
    weights_fpts: []f64 = &.{},

    /// Weight per quadrature point
    weights_qpts: []f64 = &.{},

    // ---- Face Geometry (parent space) ----

    /// Outward unit normal at each flux point, (n_fpts, n_dims)
    tnorm: Matrix(f64) = .empty,

    /// Parent-space face area scaling at each flux point
    tdA: []f64 = &.{},

    /// Orthonormal basis evaluated at the spts, (n_spts, n_spts)
    vand: Matrix(f64) = .empty,

    // ---- Operators ----

    /// Extrapolate from spts to fpts
    oppE: Matrix(f64) = .empty,

    /// Take the derivative at the solution points
    oppD: Array3(f64) = .empty,

    /// Correct the gradient at the spts using values at the fpts
    oppD_fpts: Array3(f64) = .empty,

    /// Take the divergence at the solution points
    oppDiv: Array3(f64) = .empty,

    /// Divergence of the DFR correction field, driven by normal fluxes at fpts
    oppDiv_fpts: Matrix(f64) = .empty,

    /// Interpolate from spts to plot points
    oppE_ppts: Matrix(f64) = .empty,

    /// Interpolate from spts to quadrature points
    oppE_qpts: Matrix(f64) = .empty,

    /// Build every reference-space quantity and operator. Call once, after the
    /// implementation has filled in the counts (`n_spts`, `n_fpts`, ...).
    pub fn setup(ele: *Element) Error!void {
        try ele.vtable.setLocs(ele);
        try ele.vtable.setNormals(ele);
        try ele.vtable.setVandermondeMats(ele);
        try ele.setupOperators();
        try ele.setupAuxOperators();
    }

    pub fn deinit(ele: *Element) void {
        const gpa = ele.gpa;

        ele.loc_spts.deinit(gpa);
        ele.loc_fpts.deinit(gpa);
        ele.loc_ppts.deinit(gpa);
        ele.loc_qpts.deinit(gpa);

        gpa.free(ele.weights_spts);
        gpa.free(ele.weights_fpts);
        gpa.free(ele.weights_qpts);

        ele.tnorm.deinit(gpa);
        gpa.free(ele.tdA);
        ele.vand.deinit(gpa);

        ele.oppE.deinit(gpa);
        ele.oppD.deinit(gpa);
        ele.oppD_fpts.deinit(gpa);
        ele.oppDiv.deinit(gpa);
        ele.oppDiv_fpts.deinit(gpa);
        ele.oppE_ppts.deinit(gpa);
        ele.oppE_qpts.deinit(gpa);
    }

    /// Reference-space coordinates of row `i` of a point-location matrix.
    pub fn locRow(mat: *const Matrix(f64), i: usize, n_dims: usize) []const f64 {
        return mat.data[i * mat.stride ..][0..n_dims];
    }

    /// Build the core FR operators (ZEFR `Elements::setup_FR`, tensor-product
    /// branch only).
    fn setupOperators(ele: *Element) Error!void {
        const gpa = ele.gpa;
        const n_dims = ele.n_dims;
        const n_spts = ele.n_spts;
        const n_fpts = ele.n_fpts;
        const vt = ele.vtable;

        ele.oppE = try Matrix(f64).init(gpa, n_fpts, n_spts, null);
        ele.oppD = try Array3(f64).init(gpa, n_dims, n_spts, n_spts);
        ele.oppD_fpts = try Array3(f64).init(gpa, n_dims, n_spts, n_fpts);
        ele.oppDiv = try Array3(f64).init(gpa, n_spts, n_dims, n_spts);
        ele.oppDiv_fpts = try Matrix(f64).init(gpa, n_spts, n_fpts, null);

        // Extrapolation from spts to fpts: each solution basis function
        // evaluated at each flux point.
        for (0..n_fpts) |fpt| {
            const loc = locRow(&ele.loc_fpts, fpt, n_dims);
            for (0..n_spts) |spt| {
                ele.oppE.at(fpt, spt).* = vt.calcNodalBasis(ele, spt, loc);
            }
        }

        // Gradient at the spts. oppDiv holds the same values with the dimension
        // index moved so that a divergence contracts along a row.
        for (0..n_spts) |ispt| {
            const loc = locRow(&ele.loc_spts, ispt, n_dims);
            for (0..n_dims) |dim| {
                for (0..n_spts) |jspt| {
                    const val = vt.calcDNodalBasisSpts(ele, jspt, loc, dim);
                    ele.oppD.at(dim, ispt, jspt).* = val;
                    ele.oppDiv.at(ispt, dim, jspt).* = val;
                }
            }
        }

        // Gradient at the spts of each flux point's DFR correction function.
        for (0..n_spts) |spt| {
            const loc = locRow(&ele.loc_spts, spt, n_dims);
            for (0..n_dims) |dim| {
                for (0..n_fpts) |fpt| {
                    ele.oppD_fpts.at(dim, spt, fpt).* =
                        vt.calcDNodalBasisFpts(ele, fpt, loc, dim);
                }
            }
        }

        // Divergence of the correction field, contracted with the outward normal.
        //
        // A flux point carries the *normal* flux F.n, so recovering the
        // component the divergence needs means projecting along `tnorm`. Only
        // the term normal to the flux point's own face survives: the DFR 1D grid
        // contains every solution point, so the endpoint basis function vanishes
        // at all of them and the tangential terms drop out.
        //
        // ZEFR instead multiplies the plain sum over dimensions by +/-1 from a
        // per-face table keyed on the face ordering (faces 0 and 3 in 2D, even
        // faces in 3D). Contracting with `tnorm` is equivalent -- see the
        // per-face sign test in `eles/quads_test.zig` -- and does not depend on
        // how an element happens to order its faces.
        for (0..n_dims) |dim| {
            for (0..n_fpts) |fpt| {
                const nrm = ele.tnorm.get(fpt, dim);
                for (0..n_spts) |spt| {
                    ele.oppDiv_fpts.at(spt, fpt).* += nrm * ele.oppD_fpts.get(dim, spt, fpt);
                }
            }
        }
    }

    /// Interpolation operators onto the plot and quadrature points (ZEFR
    /// `Elements::setup_aux`).
    fn setupAuxOperators(ele: *Element) Error!void {
        const gpa = ele.gpa;
        const n_dims = ele.n_dims;
        const n_spts = ele.n_spts;
        const vt = ele.vtable;

        ele.oppE_ppts = try Matrix(f64).init(gpa, ele.n_ppts, n_spts, null);
        ele.oppE_qpts = try Matrix(f64).init(gpa, ele.n_qpts, n_spts, null);

        for (0..ele.n_ppts) |ppt| {
            const loc = locRow(&ele.loc_ppts, ppt, n_dims);
            for (0..n_spts) |spt| {
                ele.oppE_ppts.at(ppt, spt).* = vt.calcNodalBasis(ele, spt, loc);
            }
        }

        for (0..ele.n_qpts) |qpt| {
            const loc = locRow(&ele.loc_qpts, qpt, n_dims);
            for (0..n_spts) |spt| {
                ele.oppE_qpts.at(qpt, spt).* = vt.calcNodalBasis(ele, spt, loc);
            }
        }
    }
};

const Config = @import("config.zig").Config;
const CellType = @import("geo.zig").CellType;

const Matrix = @import("util/matrix.zig").Matrix;
const Array3 = @import("util/array3.zig").Array3;

const std = @import("std");

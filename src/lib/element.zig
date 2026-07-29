pub const Element = struct {
    /// The VTable for any Element type.
    /// It is expected that implementations will use `@fieldParentPtr(ele)`
    pub const VTable = struct {
        /// Set all basic data, including point locations: solution, flux, plot, etc.
        setup: fn (*Element, order: u8) void,

        // set_normals: fn (std::shared_ptr<Faces> faces) void,

        // void set_oppRestart: fn (unsigned int order_restart, bool use_shape = false) ,
        set_vandermonde_mats: fn (*Element) void,

        /// Evaluate the nodal shape basis at a reference location
        eval_shape: fn (*Element, loc: []const f64, shape_vals: Matrix(f64)) void,

        /// Evaluate the derivative of the nodal shape basis at a reference location
        eval_d_shape: fn (*Element, loc: []const f64, dshape_val: Matrix(f64)) void,

        /// Evaluate the nodal basis
        eval_nodal_basis: fn (*const Element, spt: u32, loc: []const f64) f64,

        /// Compute the derivative of the nodal basis at a solution point
        eval_d_nodal_basis_spts: fn (*const Element, spt: u32, loc: []const f64, dim: u32) f64,

        /// Compute the derivative of the nodal basis at a flux point
        eval_d_nodal_basis_fpts: fn (*const Element, fpt: u32, loc: []const f64, dim: u32) f64,

        get_face_nodes: fn (*const Element, face: u32, P: u32) Matrix(f64),
        get_face_weights: fn (*const Element, face: u32, P: u32) Matrix(f64),

        project_face_point: fn (*const Element, face: u32, loc: []const f64, ploc: []f64) void,

        // virtual double eval_nodal_face_basis(unsigned int face, unsigned int pt, const double *loc) = 0;
        // virtual double eval_orthonormal_basis(unsigned int mode, const double *loc) = 0;
    };
    vtable: VTable = undefined,

    /// TODO: Pass in, don't store?
    gpa: std.mem.Allocator = undefined,

    config: *Config = undefined,

    order: u8 = 0,
    n_spts: usize = 0,
    n_fpts: usize = 0,
    n_ppts: usize = 0,
    n_qpts: usize = 0,
    n_spts_1d: usize = 0, // HACK - only applies to tensor-product elements

    n_dims: u8 = 0,
    n_faces: u8 = 0,

    // ---- Point Locations ----

    /// Solution Points
    loc_spts: Matrix(f64) = .{},

    /// Flux Points
    loc_fpts: Matrix(f64) = .{},

    /// Plot Points
    loc_ppts: Matrix(f64) = .{},

    /// Quadrature Points
    loc_qpts: Matrix(f64) = .{},

    // ---- Operators ----

    /// Take the derivative at the solution points
    oppD: Array3(f64) = .{},

    /// Take the derivative at the flux points
    oppD_fpts: Array3(f64) = .{},

    /// Extrapolate from spts to fpts
    oppE: Array3(f64) = .{},

    /// Take the divergence at the solution points
    oppDiv: Array3(f64) = .{},

    /// Take the divergence at the flux points
    oppDiv_fpts: Matrix(f64) = .{},

    /// Setup all Flux Reconstruction operators
    pub fn setupOperators(ele: *Element) !void {
        ele.oppD = .init(ele.gpa, ele.n_dims, ele.n_spts, ele.n_spts);
        ele.oppD_fpts = .init(ele.gpa, ele.n_dims, ele.n_spts, ele.n_fpts);
        ele.oppE = .init(ele.gpa, ele.n_fpts, ele.n_spts);
        ele.oppDiv = .init(ele.gpa, ele.n_spts, ele.n_dims, ele.n_spts);
        ele.oppDiv_fpts = .init(ele.gpa, ele.n_spts, ele.n_fpts);

        const n_dims = ele.n_dims;
        const n_fpts = ele.n_fpts;
        const n_spts = ele.n_spts;
        const n_spts_1d = ele.n_spts_1d;

        const loc: [3]f64 = .{ 0.0, 0.0, 0.0 };

        // Setup differentiation operator (oppD_fpts) for flux points (DFR Specific)*/
        for (0..n_dims) |dim| {
            for (0..n_fpts) |fpt| {
                for (0..n_spts) |spt| {
                    for (0..n_dims) |d| {
                        loc[d] = ele.loc_spts(spt, d);
                    }
                    ele.oppD_fpts.at(dim, spt, fpt).* = ele.vtable.calc_d_nodal_basis_fpts(ele, fpt, loc, dim);
                }
            }
        }

        // Setup divergence operator (oppDiv) for solution pointsg
        // Note: This is essentially the same as oppD, but with dimensions oriented in a rowg
        for (0..n_dims) |dim| {
            for (0..n_spts) |jspt| {
                for (0..n_spts) |ispt| {
                    for (0..n_dims) |d| {
                        loc[d] = ele.loc_spts(ispt, d);
                    }
                    ele.oppDiv.at(ispt, dim, jspt).* = ele.vtable.calc_d_nodal_basis_spts(ele, jspt, loc, dim);
                }
            }
        }

        // Setup divergence operator (oppDiv_fpts) for flux points by combining dimensions of oppD_fptsg
        for (0..n_dims) |dim| {
            for (0..n_fpts) |fpt| {
                // TODO: This block only applies to tensor-product elements
                // Set positive parent sign convention into operator based on faceg
                var fac: i32 = 1;
                if (n_dims == 2) {
                    const face = @divTrunc(fpt, n_spts_1d);
                    if (face == 0 or face == 3) // Bottom and Left face
                        fac = -1;
                } else if (n_dims == 3) {
                    const face = @divTrunc(fpt, (n_spts_1d * n_spts_1d));
                    if (@mod(face, 2) == 0) // Bottom, Left, and Front face
                        fac = -1;
                }

                for (0..n_spts) |spt| {
                    ele.oppDiv_fpts.at(spt, fpt).* += fac * ele.oppD_fpts.get(dim, spt, fpt);
                }
            }
        }
    }
};

const Config = struct {};
// const Config = @import("config.zig").Config;

const Matrix = @import("matrix.zig").Matrix;
const Array3 = @import("array3.zig").Array3;
const Array4 = @import("array4.zig").Array4;

const std = @import("std");

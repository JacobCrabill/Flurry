//! Core Flux Reconstruction solver
const std = @import("std");

const array3 = @import("util/array3.zig");
const array4 = @import("util/array4.zig");

pub const Solver = struct {
    // ---- Solution Variables ----

    /// Solution at the solution points
    u_spts: Array3d,
    /// Solution at the flux points
    u_fpts: Array3d,

    /// Derivative of the solution at the flux points
    du_fpts: Array3d,
    /// Derivative of the solution at the solution points
    du_spts: Array3d,

    /// Derivative of the corrected solution at the flux points (Viscous cases)
    du_c_fpts: Array3d,

    // ---- Flux Variables ----

    /// Flux at the solution points
    f_spts: Array4d,
    /// Flux at the flux points
    f_fpts: Array4d,
    /// Normal flux at the flux points
    fn_fpts: Array3d,

    /// Derivative of the flux at the solution points (One per dimension)
    df_spts: std.ArrayList(Array3d),
    /// Divergence of the flux at the solution points (One per RK step)
    divf_spts: std.ArrayList(Array3d),

    // ---- Plotting Variables ----

    /// Primitive variables at the solution points (Used for plotting)
    v_spts: Array3d,
    /// Primitive variables at the plot points (Used for plotting)
    v_ppts: Array3d,
};

const Array3d = array3.Array3d;
const Array3f = array3.Array3f;
const Array3u = array3.Array3u;
const Array3i = array3.Array3i;
const Array3z = array3.Array3z;

const Array4d = array4.Array4d;
const Array4f = array4.Array4f;
const Array4u = array4.Array4u;
const Array4i = array4.Array4i;
const Array4z = array4.Array4z;

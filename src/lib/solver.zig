//! Core Flux Reconstruction solver.
//!
//! Owns the per-cell solution, flux and geometry arrays, and drives the DFR
//! residual using the reference-element operators from `element.zig`:
//!
//!   1. `extrapolateU`      U_spts  -> U_fpts       via oppE
//!   2. faces: scatter, apply BCs, compute the common normal flux
//!   3. `computeFluxSpts`   U_spts  -> F_spts       physical flux, then
//!                                                  transformed to reference space
//!   4. `computeDivFSpts`   F_spts  -> divF_spts    via oppDiv
//!   5. `computeDivFFpts`   Fcomm   -> divF_spts   += via oppDiv_fpts
//!   6. `rkStage`           U_spts  -= a dt / |J| divF_spts
//!
//! Array layout follows ZEFR: the element index is last and therefore
//! contiguous, so every operator application is one dense matrix product with
//! `N = n_vars * n_eles`.
//!
//! Not ported: implicit time stepping, p-multigrid, overset, moving grids,
//! filtering, MPI.

pub const Error = error{
    OutOfMemory,
    UnsupportedOrder,
    UnsupportedShapeOrder,
    /// The mesh mixes cell types, or uses one the solver has no element for
    UnsupportedCellType,
    /// `core.n_dims` is neither 2 nor 3
    UnsupportedDimension,
    /// A cell's mapping is inverted or degenerate
    NegativeJacobian,
    /// `mesh.setupGlobalFpts` was not run, or was run for a different order
    ConnectivityNotProcessed,
    /// `l2Error` needs a quadrature rule, which `test_case.n_qpts_1d` sets and
    /// `Loader.initialize` zeroes when `output.error_freq` is 0
    NoQuadraturePoints,
    /// The configured test case has no exact solution, or `err_field` names a
    /// variable this equation set does not have
    NoExactSolution,
    NotImplemented,
} || faces_mod.Error || testcase.Error || gpu.Error;

/// Explicit Runge-Kutta tableau.
///
/// `alpha` advances the intermediate stages, `beta` combines the stage
/// residuals into the final update, and `c` places each stage in time.
pub const RkScheme = struct {
    n_stages: usize,
    alpha: []const f64,
    beta: []const f64,
    c: []const f64,

    /// Whether the final update is a weighted sum over all stage residuals
    /// (a full Butcher tableau) rather than just the last stage.
    combines_stages: bool,

    pub const euler: RkScheme = .{
        .n_stages = 1,
        .alpha = &.{},
        .beta = &.{1.0},
        .c = &.{0.0},
        .combines_stages = true,
    };

    pub const rk44: RkScheme = .{
        .n_stages = 4,
        .alpha = &.{ 0.5, 0.5, 1.0 },
        .beta = &.{ 1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0 },
        .c = &.{ 0.0, 0.5, 0.5, 1.0 },
        .combines_stages = true,
    };

    /// Jameson-style RK: every stage advances from the start of the step by
    /// `alpha[stage]`, overwriting the solution, so there is no final
    /// combination.
    ///
    /// Stage `i` evaluates its residual on what stage `i-1` produced, which sits
    /// at `t + alpha[i-1] dt` -- so `c` is `alpha` shifted, starting at zero.
    /// ZEFR sets `rk_c = rk_alpha` outright, which places the first residual at
    /// `t + 0.153 dt` even though the solution there is still the one from `t`.
    /// That only shows up through time-dependent sources or boundary data.
    pub const rk_jameson: RkScheme = .{
        .n_stages = 4,
        .alpha = &.{ 0.153, 0.442, 0.930, 1.0 },
        .beta = &.{},
        .c = &.{ 0.0, 0.153, 0.442, 0.930 },
        .combines_stages = false,
    };

    pub fn fromConfig(scheme: cfg.DtScheme) Error!RkScheme {
        return switch (scheme) {
            .euler => euler,
            .rk44 => rk44,
            .rkJ => rk_jameson,
            // RK54 needs the low-storage two-register update, and Steady needs
            // an implicit solve. Neither is ported.
            .rk54, .steady => error.NotImplemented,
        };
    }
};

pub const Solver = struct {
    gpa: std.mem.Allocator,
    config: *const cfg.Config,
    mesh: *const Geo,

    params: flux.FlowParams,
    rk: RkScheme,

    /// The reference element. One type for now; a mixed mesh needs one of
    /// these per cell type.
    quad: Quad,
    faces: Faces,

    n_dims: usize = 0,
    n_vars: usize = 0,
    n_eles: usize = 0,

    current_iter: u32 = 0,
    flow_time: f64 = 0.0,
    dt: f64 = 0.0,

    /// Optional GPU back end. When set, the operators that have been ported run
    /// there and the rest stay on the CPU; when null everything is on the CPU,
    /// which is the reference the GPU path is checked against. Must outlive the
    /// solver.
    device: ?*gpu.Device = null,

    // ---- Solution Variables ----

    /// Solution at the solution points, (spt, var, ele)
    u_spts: Array3(f64) = .empty,
    /// Solution at the start of the current step, for the RK combination
    u_ini: Array3(f64) = .empty,
    /// Solution at the flux points, (fpt, var, ele)
    u_fpts: Array3(f64) = .empty,

    /// Gradient of the solution at the solution points, (dim, spt, var, ele).
    /// Reference-space until `computeFluxSpts` converts it in place.
    du_spts: Array4(f64) = .empty,
    /// Gradient at the flux points, (dim, fpt, var, ele)
    du_fpts: Array4(f64) = .empty,

    // ---- Flux Variables ----

    /// Reference-space flux at the solution points, (dim, spt, var, ele)
    f_spts: Array4(f64) = .empty,

    /// Common normal flux gathered back from the faces, (fpt, var, ele)
    f_comm: Array3(f64) = .empty,
    /// Common solution gathered back from the faces, (fpt, var, ele)
    u_comm: Array3(f64) = .empty,

    /// Divergence of the flux at the solution points, one per RK stage:
    /// (stage, spt, var, ele) flattened into an Array4
    divf_spts: Array4(f64) = .empty,

    // ---- Geometry ----

    /// Physical coordinates of each cell's shape nodes, (node, dim, ele)
    nodes: Array3(f64) = .empty,

    /// Mapping Jacobian at the solution points, (dim_ref, spt, dim_phys, ele)
    jaco_spts: Array4(f64) = .empty,
    /// Adjugate (inverse times determinant) at the spts, same layout
    inv_jaco_spts: Array4(f64) = .empty,
    /// Jacobian determinant at the solution points, (spt, ele)
    jaco_det_spts: Matrix(f64) = .empty,

    /// Adjugate at the flux points, (dim_ref, fpt, dim_phys, ele)
    inv_jaco_fpts: Array4(f64) = .empty,

    /// Outward *unit* normal at each flux point, (fpt, dim, ele)
    norm_fpts: Array3(f64) = .empty,

    /// Physical-to-reference face scaling at each flux point, (fpt, ele).
    /// The magnitude of the un-normalized transformed normal, i.e. the ratio of
    /// physical to reference face measure.
    d_a_fpts: Matrix(f64) = .empty,

    /// Cell volumes
    vol: []f64 = &.{},

    /// Physical coordinates of the solution points, (spt, dim, ele)
    coord_spts: Array3(f64) = .empty,

    /// Physical coordinates of the flux points, (fpt, dim, ele)
    coord_fpts: Array3(f64) = .empty,

    /// Create a solver for a mesh that has already been read, had its
    /// connectivity processed, and had its global flux points laid out:
    ///
    ///     try mesh.createMesh();            // or readGmsh
    ///     try mesh.processConnectivity();
    ///     try mesh.setupGlobalFpts(config.core.order + 1);
    ///     var solver = try Solver.init(gpa, &config, &mesh);
    ///
    /// `mesh` and `config` must outlive the solver.
    pub fn init(
        gpa: std.mem.Allocator,
        config: *const cfg.Config,
        mesh: *const Geo,
    ) Error!Solver {
        const n_dims: usize = config.core.n_dims;
        if (n_dims != 2) {
            // Hex elements are not implemented yet, so 3D cannot be set up.
            return error.UnsupportedDimension;
        }
        for (mesh.ctype.items) |ct| {
            if (ct != .quad) return error.UnsupportedCellType;
        }

        const params = flux.FlowParams.fromConfig(config);
        const n_vars = flux.nVars(config.equation.equation, n_dims);
        const n_eles = mesh.n_eles;
        const order = config.core.order;
        const n_nodes = if (n_eles > 0) mesh.c2nv.items[0] else 4;

        var s: Solver = .{
            .gpa = gpa,
            .config = config,
            .mesh = mesh,
            .params = params,
            .rk = try RkScheme.fromConfig(config.time.dt_scheme),
            .quad = Quad.init(gpa, config, order, n_nodes),
            // Replaced below, once the element's flux point count is known.
            .faces = try Faces.init(gpa, config, params, 0, 0),
            .n_dims = n_dims,
            .n_vars = n_vars,
            .n_eles = n_eles,
            .dt = config.time.dt orelse 0.0,
        };
        errdefer s.deinit();

        try s.quad.ele.setup();

        // The caller owns mesh setup; check that its global flux point layout
        // was built for this element.
        if (mesh.n_fpts_per_face != s.quad.ele.n_fpts_per_face) {
            return error.ConnectivityNotProcessed;
        }

        s.faces.deinit();
        s.faces = try Faces.init(gpa, config, params, mesh.n_gfpts, mesh.n_gfpts_bnd);
        // Borrowed from the mesh, which outlives the solver
        s.faces.gfpt2bnd = mesh.gfpt2bnd.items;
        s.faces.bc_list = mesh.bc_list.items;

        try s.allocate();
        try s.computeTransforms();
        s.setFaceGeometry();

        return s;
    }

    pub fn deinit(s: *Solver) void {
        const gpa = s.gpa;

        s.u_spts.deinit(gpa);
        s.u_ini.deinit(gpa);
        s.u_fpts.deinit(gpa);
        s.du_spts.deinit(gpa);
        s.du_fpts.deinit(gpa);
        s.f_spts.deinit(gpa);
        s.f_comm.deinit(gpa);
        s.u_comm.deinit(gpa);
        s.divf_spts.deinit(gpa);

        s.nodes.deinit(gpa);
        s.jaco_spts.deinit(gpa);
        s.inv_jaco_spts.deinit(gpa);
        s.jaco_det_spts.deinit(gpa);
        s.inv_jaco_fpts.deinit(gpa);
        s.norm_fpts.deinit(gpa);
        s.d_a_fpts.deinit(gpa);
        gpa.free(s.vol);
        s.coord_spts.deinit(gpa);
        s.coord_fpts.deinit(gpa);

        s.faces.deinit();
        s.quad.deinit();
    }

    fn allocate(s: *Solver) Error!void {
        const gpa = s.gpa;
        const ele = &s.quad.ele;
        const nv = s.n_vars;
        const ne = s.n_eles;
        const nd = s.n_dims;

        s.u_spts = try Array3(f64).init(gpa, ele.n_spts, nv, ne);
        s.u_fpts = try Array3(f64).init(gpa, ele.n_fpts, nv, ne);
        s.f_spts = try Array4(f64).init(gpa, nd, ele.n_spts, nv, ne);
        s.f_comm = try Array3(f64).init(gpa, ele.n_fpts, nv, ne);
        s.divf_spts = try Array4(f64).init(gpa, s.rk.n_stages, ele.n_spts, nv, ne);

        // u_ini is only needed when a later stage has to restart from the
        // beginning of the step.
        if (s.rk.n_stages > 1) {
            s.u_ini = try Array3(f64).init(gpa, ele.n_spts, nv, ne);
        }

        if (s.config.equation.viscous) {
            s.du_spts = try Array4(f64).init(gpa, nd, ele.n_spts, nv, ne);
            s.du_fpts = try Array4(f64).init(gpa, nd, ele.n_fpts, nv, ne);
            s.u_comm = try Array3(f64).init(gpa, ele.n_fpts, nv, ne);
        }

        s.nodes = try Array3(f64).init(gpa, ele.n_nodes, nd, ne);
        s.jaco_spts = try Array4(f64).init(gpa, nd, ele.n_spts, nd, ne);
        s.inv_jaco_spts = try Array4(f64).init(gpa, nd, ele.n_spts, nd, ne);
        s.jaco_det_spts = try Matrix(f64).init(gpa, ele.n_spts, ne, null);
        s.inv_jaco_fpts = try Array4(f64).init(gpa, nd, ele.n_fpts, nd, ne);
        s.norm_fpts = try Array3(f64).init(gpa, ele.n_fpts, nd, ne);
        s.d_a_fpts = try Matrix(f64).init(gpa, ele.n_fpts, ne, null);
        s.coord_spts = try Array3(f64).init(gpa, ele.n_spts, nd, ne);
        s.coord_fpts = try Array3(f64).init(gpa, ele.n_fpts, nd, ne);
        s.vol = try gpa.alloc(f64, ne);
        @memset(s.vol, 0.0);
    }

    // ---- Geometry ----

    /// Gather each cell's shape nodes, then evaluate the mapping Jacobian, its
    /// adjugate and determinant at the solution and flux points.
    ///
    /// ZEFR does the Jacobian as one big `dshape * nodes` matrix product; here
    /// the shape derivatives are evaluated per point instead, which keeps the
    /// element's `calcDShape` as the single definition of the mapping.
    pub fn computeTransforms(s: *Solver) Error!void {
        const ele = &s.quad.ele;
        const nd = s.n_dims;
        const mesh = s.mesh;

        // Shape nodes, from the mesh
        for (0..s.n_eles) |e| {
            for (0..ele.n_nodes) |node| {
                const iv = mesh.c2v.get(e, node);
                for (0..nd) |d| s.nodes.at(node, d, e).* = mesh.xv.get(iv, d);
            }
        }

        var dshape = try Matrix(f64).init(s.gpa, ele.n_nodes, nd, null);
        defer dshape.deinit(s.gpa);
        const shape = try s.gpa.alloc(f64, ele.n_nodes);
        defer s.gpa.free(shape);

        // Solution points: Jacobian, adjugate, determinant, and coordinates
        for (0..ele.n_spts) |spt| {
            const loc = Element.locRow(&ele.loc_spts, spt, nd);
            try ele.vtable.calcDShape(ele, loc, &dshape);
            try ele.vtable.calcShape(ele, loc, shape);

            for (0..s.n_eles) |e| {
                for (0..nd) |dr| {
                    for (0..nd) |dp| {
                        var sum: f64 = 0.0;
                        for (0..ele.n_nodes) |node| {
                            sum += dshape.get(node, dr) * s.nodes.get(node, dp, e);
                        }
                        s.jaco_spts.at(dr, spt, dp, e).* = sum;
                    }
                }
                for (0..nd) |dp| {
                    var sum: f64 = 0.0;
                    for (0..ele.n_nodes) |node| {
                        sum += shape[node] * s.nodes.get(node, dp, e);
                    }
                    s.coord_spts.at(spt, dp, e).* = sum;
                }
            }
        }
        try s.setInverseTransforms(&s.jaco_spts, &s.inv_jaco_spts, &s.jaco_det_spts, ele.n_spts);

        // Flux points: only the adjugate is needed, to rotate the reference
        // normals into physical space, so the Jacobian itself is scratch.
        var jaco_fpts = try Array4(f64).init(s.gpa, nd, ele.n_fpts, nd, s.n_eles);
        defer jaco_fpts.deinit(s.gpa);

        for (0..ele.n_fpts) |fpt| {
            const loc = Element.locRow(&ele.loc_fpts, fpt, nd);
            try ele.vtable.calcDShape(ele, loc, &dshape);
            try ele.vtable.calcShape(ele, loc, shape);

            for (0..s.n_eles) |e| {
                for (0..nd) |dr| {
                    for (0..nd) |dp| {
                        var sum: f64 = 0.0;
                        for (0..ele.n_nodes) |node| {
                            sum += dshape.get(node, dr) * s.nodes.get(node, dp, e);
                        }
                        jaco_fpts.at(dr, fpt, dp, e).* = sum;
                    }
                }
                for (0..nd) |dp| {
                    var sum: f64 = 0.0;
                    for (0..ele.n_nodes) |node| {
                        sum += shape[node] * s.nodes.get(node, dp, e);
                    }
                    s.coord_fpts.at(fpt, dp, e).* = sum;
                }
            }
        }
        try s.setInverseTransforms(&jaco_fpts, &s.inv_jaco_fpts, null, ele.n_fpts);
        s.computeFaceNormals();

        // Cell volumes, by the solution-point quadrature
        for (0..s.n_eles) |e| {
            var v: f64 = 0.0;
            for (0..ele.n_spts) |spt| {
                v += ele.weights_spts[spt] * s.jaco_det_spts.get(spt, e);
            }
            s.vol[e] = v;
        }
    }

    /// Rotate the reference-space face normals into physical space.
    ///
    /// `norm = adj^T . tnorm` is the un-normalized outward normal; its magnitude
    /// is the physical/reference face measure ratio `dA`. Splitting them this
    /// way means a normal flux is `(F . norm_unit) * dA`, which is exactly what
    /// `oppDiv_fpts` consumes.
    fn computeFaceNormals(s: *Solver) void {
        const ele = &s.quad.ele;
        const nd = s.n_dims;

        for (0..s.n_eles) |e| {
            for (0..ele.n_fpts) |fpt| {
                var norm: [3]f64 = .{ 0, 0, 0 };
                for (0..nd) |d1| {
                    for (0..nd) |d2| {
                        norm[d1] += s.inv_jaco_fpts.get(d2, fpt, d1, e) * ele.tnorm.get(fpt, d2);
                    }
                }

                var mag: f64 = 0.0;
                for (0..nd) |d| mag += norm[d] * norm[d];
                mag = @sqrt(mag);
                s.d_a_fpts.at(fpt, e).* = mag;

                // A collapsed face has zero measure and no meaningful normal.
                const inv = if (mag > 0.0) 1.0 / mag else 0.0;
                for (0..nd) |d| s.norm_fpts.at(fpt, d, e).* = norm[d] * inv;
            }
        }
    }

    /// Adjugate (inverse times determinant) and, optionally, determinant of a
    /// stored Jacobian field. Safe to alias `jaco` and `inv_jaco`.
    fn setInverseTransforms(
        s: *Solver,
        jaco: *const Array4(f64),
        inv_jaco: *Array4(f64),
        jaco_det: ?*Matrix(f64),
        n_pts: usize,
    ) Error!void {
        for (0..s.n_eles) |e| {
            for (0..n_pts) |pt| {
                switch (s.n_dims) {
                    2 => {
                        const xr = jaco.get(0, pt, 0, e);
                        const xs = jaco.get(1, pt, 0, e);
                        const yr = jaco.get(0, pt, 1, e);
                        const ys = jaco.get(1, pt, 1, e);

                        const det = xr * ys - xs * yr;
                        if (det <= 0.0) return error.NegativeJacobian;
                        if (jaco_det) |jd| jd.at(pt, e).* = det;

                        // adj[i][j] = |J| dxi_i/dx_j, so the off-diagonals are
                        // -dx/deta and -dy/dxi -- not each other's transpose.
                        inv_jaco.at(0, pt, 0, e).* = ys;
                        inv_jaco.at(0, pt, 1, e).* = -xs;
                        inv_jaco.at(1, pt, 0, e).* = -yr;
                        inv_jaco.at(1, pt, 1, e).* = xr;
                    },
                    3 => {
                        const xr = jaco.get(0, pt, 0, e);
                        const xs = jaco.get(1, pt, 0, e);
                        const xt = jaco.get(2, pt, 0, e);
                        const yr = jaco.get(0, pt, 1, e);
                        const ys = jaco.get(1, pt, 1, e);
                        const yt = jaco.get(2, pt, 1, e);
                        const zr = jaco.get(0, pt, 2, e);
                        const zs = jaco.get(1, pt, 2, e);
                        const zt = jaco.get(2, pt, 2, e);

                        const det = xr * (ys * zt - yt * zs) -
                            xs * (yr * zt - yt * zr) +
                            xt * (yr * zs - ys * zr);
                        if (det <= 0.0) return error.NegativeJacobian;
                        if (jaco_det) |jd| jd.at(pt, e).* = det;

                        inv_jaco.at(0, pt, 0, e).* = ys * zt - yt * zs;
                        inv_jaco.at(0, pt, 1, e).* = xt * zs - xs * zt;
                        inv_jaco.at(0, pt, 2, e).* = xs * yt - xt * ys;
                        inv_jaco.at(1, pt, 0, e).* = yt * zr - yr * zt;
                        inv_jaco.at(1, pt, 1, e).* = xr * zt - xt * zr;
                        inv_jaco.at(1, pt, 2, e).* = xt * yr - xr * yt;
                        inv_jaco.at(2, pt, 0, e).* = yr * zs - ys * zr;
                        inv_jaco.at(2, pt, 1, e).* = xs * zr - xr * zs;
                        inv_jaco.at(2, pt, 2, e).* = xr * ys - xs * yr;
                    },
                    else => return error.UnsupportedDimension,
                }
            }
        }
    }

    // ---- Initialization ----

    /// Apply the initial condition `test_case.test_case` selects.
    ///
    /// The solution is *collocated* at the solution points, not projected onto
    /// the basis. That is what ZEFR does and it costs one order in the initial
    /// error, which does not change the asymptotic rate.
    pub fn initializeU(s: *Solver) Error!void {
        const ele = &s.quad.ele;
        const tc = try testcase.TestCase.fromConfig(s.config);

        if (!tc.isAnalytic()) {
            const state = s.params.freestreamState(2, s.config.equation.equation);
            for (0..ele.n_spts) |spt| {
                for (0..s.n_vars) |n| {
                    for (0..s.n_eles) |e| s.u_spts.at(spt, n, e).* = state[n];
                }
            }
            return;
        }

        const bounds = s.meshBounds();
        for (0..ele.n_spts) |spt| {
            for (0..s.n_eles) |e| {
                const state = testcase.exactState(
                    tc,
                    s.params,
                    s.coord_spts.get(spt, 0, e),
                    s.coord_spts.get(spt, 1, e),
                    0.0,
                    .{ bounds[0], bounds[1] },
                );
                for (0..s.n_vars) |n| s.u_spts.at(spt, n, e).* = state[n];
            }
        }
    }

    /// L2 norm of the error in variable `test_case.err_field`, against the
    /// exact solution at the current `flow_time`.
    ///
    /// Measured at a separate Gauss-Legendre rule (`test_case.n_qpts_1d`)
    /// rather than at the solution points: the scheme is exact *at* its own
    /// collocation points to within its own consistency, so sampling there
    /// flatters it. `Loader.initialize` zeroes `n_qpts_1d` when `error_freq` is
    /// 0, so a case that never asks for error pays nothing for the points.
    ///
    /// Normalized by the domain volume, making it an RMS error: the value is
    /// comparable across meshes, which is the whole point of a refinement study.
    pub fn l2Error(s: *const Solver, gpa: std.mem.Allocator) Error!f64 {
        const ele = &s.quad.ele;
        if (ele.n_qpts == 0) return error.NoQuadraturePoints;

        const tc = try testcase.TestCase.fromConfig(s.config);
        if (!tc.isAnalytic()) return error.NoExactSolution;

        const n = s.config.test_case.err_field;
        if (n >= s.n_vars) return error.NoExactSolution;

        var u_qpts = try Array3(f64).init(gpa, ele.n_qpts, s.n_vars, s.n_eles);
        defer u_qpts.deinit(gpa);
        gemm(
            ele.n_qpts,
            s.n_vars * s.n_eles,
            ele.n_spts,
            ele.oppE_qpts.data,
            s.u_spts.data,
            u_qpts.data,
            .overwrite,
        );

        const nd = s.n_dims;
        const bounds = s.meshBounds();

        var dshape = try Matrix(f64).init(gpa, ele.n_nodes, nd, null);
        defer dshape.deinit(gpa);
        const shape = try gpa.alloc(f64, ele.n_nodes);
        defer gpa.free(shape);

        var sq_error: f64 = 0.0;
        var volume: f64 = 0.0;

        for (0..ele.n_qpts) |qpt| {
            const loc = Element.locRow(&ele.loc_qpts, qpt, nd);
            try ele.vtable.calcShape(ele, loc, shape);
            try ele.vtable.calcDShape(ele, loc, &dshape);

            for (0..s.n_eles) |e| {
                var coord: [2]f64 = .{ 0, 0 };
                for (0..nd) |d| {
                    var sum: f64 = 0.0;
                    for (0..ele.n_nodes) |node| sum += shape[node] * s.nodes.get(node, d, e);
                    coord[d] = sum;
                }

                // |J| at the quadrature point, so the integral is over physical
                // space rather than reference space
                var jaco: [2][2]f64 = .{ .{ 0, 0 }, .{ 0, 0 } };
                for (0..nd) |dr| {
                    for (0..nd) |dp| {
                        var sum: f64 = 0.0;
                        for (0..ele.n_nodes) |node| {
                            sum += dshape.get(node, dr) * s.nodes.get(node, dp, e);
                        }
                        jaco[dr][dp] = sum;
                    }
                }
                const det = jaco[0][0] * jaco[1][1] - jaco[0][1] * jaco[1][0];

                const exact = testcase.exactState(
                    tc,
                    s.params,
                    coord[0],
                    coord[1],
                    s.flow_time,
                    .{ bounds[0], bounds[1] },
                );

                const err = exact[n] - u_qpts.get(qpt, n, e);
                const w = ele.weights_qpts[qpt] * det;
                sq_error += w * err * err;
                volume += w;
            }
        }

        return @sqrt(sq_error / volume);
    }

    /// Bounding box of the mesh nodes, as `[dim][lo, hi]`.
    pub fn meshBounds(s: *const Solver) [3][2]f64 {
        var out: [3][2]f64 = @splat(.{ std.math.inf(f64), -std.math.inf(f64) });
        for (0..s.n_eles) |e| {
            for (0..s.quad.ele.n_nodes) |node| {
                for (0..s.n_dims) |d| {
                    const v = s.nodes.get(node, d, e);
                    out[d][0] = @min(out[d][0], v);
                    out[d][1] = @max(out[d][1], v);
                }
            }
        }
        for (s.n_dims..3) |d| out[d] = .{ 0, 0 };
        return out;
    }

    // ---- Operator applications ----

    /// U at the solution points -> U at the flux points.
    ///
    /// The first operator to have a GPU path: with `device` set it dispatches
    /// spock's dgemm instead of running `gemm` here. Everything else in the
    /// step still runs on the CPU, which works only because the device copies
    /// its operands in and out around the dispatch -- see `gpu.zig`.
    pub fn extrapolateU(s: *Solver) Error!void {
        const ele = &s.quad.ele;
        if (s.device) |dev| {
            return dev.gemmHost(
                ele.n_fpts,
                s.n_vars * s.n_eles,
                ele.n_spts,
                ele.oppE.data,
                s.u_spts.data,
                s.u_fpts.data,
                .overwrite,
            );
        }
        gemm(
            ele.n_fpts,
            s.n_vars * s.n_eles,
            ele.n_spts,
            ele.oppE.data,
            s.u_spts.data,
            s.u_fpts.data,
            .overwrite,
        );
    }

    /// U at the solution points -> U at the equispaced plot points, for output.
    ///
    /// `out` is `(ppt, var, ele)` and must already be that size. Kept off the
    /// solver's own arrays because this runs at `write_freq`, not every step.
    pub fn extrapolateToPpts(s: *const Solver, out: *Array3(f64)) void {
        const ele = &s.quad.ele;
        gemm(
            ele.n_ppts,
            s.n_vars * s.n_eles,
            ele.n_spts,
            ele.oppE_ppts.data,
            s.u_spts.data,
            out.data,
            .overwrite,
        );
    }

    /// Physical coordinates of the plot points, `(ppt, dim, ele)`.
    ///
    /// The same mapping `computeTransforms` applies at the solution and flux
    /// points, evaluated where the output needs it instead of being carried
    /// around for the whole run.
    pub fn plotPointCoords(s: *const Solver, gpa: std.mem.Allocator, out: *Array3(f64)) Error!void {
        const ele = &s.quad.ele;
        const nd = s.n_dims;

        const shape = try gpa.alloc(f64, ele.n_nodes);
        defer gpa.free(shape);

        for (0..ele.n_ppts) |ppt| {
            const loc = Element.locRow(&ele.loc_ppts, ppt, nd);
            try ele.vtable.calcShape(ele, loc, shape);

            for (0..s.n_eles) |e| {
                for (0..nd) |d| {
                    var sum: f64 = 0.0;
                    for (0..ele.n_nodes) |node| sum += shape[node] * s.nodes.get(node, d, e);
                    out.at(ppt, d, e).* = sum;
                }
            }
        }
    }

    /// Reference-space gradient contribution from the solution points.
    pub fn computeGradSpts(s: *Solver) void {
        const ele = &s.quad.ele;
        gemm(
            ele.n_spts * s.n_dims,
            s.n_vars * s.n_eles,
            ele.n_spts,
            ele.oppD.data,
            s.u_spts.data,
            s.du_spts.data,
            .overwrite,
        );
    }

    /// Gradient correction from the common solution at the flux points.
    pub fn computeGradFpts(s: *Solver) void {
        const ele = &s.quad.ele;
        gemm(
            ele.n_spts * s.n_dims,
            s.n_vars * s.n_eles,
            ele.n_fpts,
            ele.oppD_fpts.data,
            s.u_comm.data,
            s.du_spts.data,
            .accumulate,
        );
    }

    /// Divergence contribution from the flux at the solution points.
    pub fn computeDivFSpts(s: *Solver, stage: usize) void {
        const ele = &s.quad.ele;
        const per_stage = ele.n_spts * s.n_vars * s.n_eles;
        gemm(
            ele.n_spts,
            s.n_vars * s.n_eles,
            ele.n_spts * s.n_dims,
            ele.oppDiv.data,
            s.f_spts.data,
            s.divf_spts.data[stage * per_stage ..][0..per_stage],
            .overwrite,
        );
    }

    /// Divergence correction from the common normal flux at the flux points.
    pub fn computeDivFFpts(s: *Solver, stage: usize) void {
        const ele = &s.quad.ele;
        const per_stage = ele.n_spts * s.n_vars * s.n_eles;
        gemm(
            ele.n_spts,
            s.n_vars * s.n_eles,
            ele.n_fpts,
            ele.oppDiv_fpts.data,
            s.f_comm.data,
            s.divf_spts.data[stage * per_stage ..][0..per_stage],
            .accumulate,
        );
    }

    // ---- Physical flux ----

    /// Evaluate the physical flux at every solution point and transform it into
    /// reference space. For viscous runs the reference-space gradient in
    /// `du_spts` is converted to a physical gradient in place first.
    pub fn computeFluxSpts(s: *Solver) void {
        switch (s.config.equation.equation) {
            .adv_diff => s.fluxSpts(2, .adv_diff),
            .euler_ns => s.fluxSpts(2, .euler_ns),
        }
    }

    fn fluxSpts(s: *Solver, comptime nd: usize, comptime equation: cfg.Equation) void {
        const ele = &s.quad.ele;
        const n_vars = comptime flux.nVars(equation, nd);
        const viscous = s.config.equation.viscous;

        for (0..ele.n_spts) |spt| {
            for (0..s.n_eles) |e| {
                var u: [nd + 2]f64 = @splat(0.0);
                for (0..n_vars) |n| u[n] = s.u_spts.get(spt, n, e);

                // Metric terms: the adjugate, and 1/det for the gradient
                var adj: [nd][nd]f64 = undefined;
                for (0..nd) |d1| {
                    for (0..nd) |d2| adj[d1][d2] = s.inv_jaco_spts.get(d1, spt, d2, e);
                }

                var du: [nd + 2][nd]f64 = @splat(@splat(0.0));
                if (viscous) {
                    const inv_det = 1.0 / s.jaco_det_spts.get(spt, e);
                    for (0..n_vars) |n| {
                        for (0..nd) |d1| {
                            var sum: f64 = 0.0;
                            for (0..nd) |d2| {
                                sum += s.du_spts.get(d2, spt, n, e) * adj[d2][d1];
                            }
                            du[n][d1] = sum * inv_det;
                            // Publish the physical gradient: the viscous face
                            // flux and any gradient output both want it.
                            s.du_spts.at(d1, spt, n, e).* = du[n][d1];
                        }
                    }
                }

                // Physical flux
                var f: [nd + 2][nd]f64 = @splat(@splat(0.0));
                switch (equation) {
                    .adv_diff => {
                        const fc = flux.convAdvDiff(nd, .{u[0]}, s.params);
                        f[0] = fc[0];
                        if (viscous) {
                            var f1: [1][nd]f64 = .{f[0]};
                            flux.viscAdvDiffAdd(nd, .{du[0]}, &f1, s.params);
                            f[0] = f1[0];
                        }
                    },
                    .euler_ns => {
                        const fc = flux.convEulerNS(nd, u, s.params);
                        f = fc.f;
                        if (viscous) flux.viscEulerNSAdd(nd, u, du, &f, s.params);
                    },
                }

                // Transform to reference space: tF = adj . F
                for (0..n_vars) |n| {
                    for (0..nd) |d1| {
                        var sum: f64 = 0.0;
                        for (0..nd) |d2| sum += f[n][d2] * adj[d1][d2];
                        s.f_spts.at(d1, spt, n, e).* = sum;
                    }
                }
            }
        }
    }

    // ---- Residual ----

    /// One residual evaluation: fills `divf_spts[stage]`.
    pub fn computeResidual(s: *Solver, stage: usize) Error!void {
        try s.extrapolateU();
        s.scatterUToFaces();

        // STUB: the right-hand state of every boundary flux point. See faces.zig.
        try s.faces.applyBcs();

        if (s.config.equation.viscous) {
            s.computeGradSpts();
            s.faces.computeCommonU();
            s.gatherCommonUFromFaces();
            s.computeGradFpts();
        }

        s.computeFluxSpts();

        if (s.config.equation.viscous) {
            s.extrapolateGrad();
            s.scatterGradToFaces();
            try s.faces.applyBcsGrad();
        }

        s.computeDivFSpts(stage);
        s.faces.computeCommonF();
        s.gatherCommonFFromFaces();
        s.computeDivFFpts(stage);
    }

    /// Physical solution gradient at the solution points -> flux points.
    pub fn extrapolateGrad(s: *Solver) void {
        const ele = &s.quad.ele;
        const per_dim = ele.n_spts * s.n_vars * s.n_eles;
        const per_dim_f = ele.n_fpts * s.n_vars * s.n_eles;
        for (0..s.n_dims) |dim| {
            gemm(
                ele.n_fpts,
                s.n_vars * s.n_eles,
                ele.n_spts,
                ele.oppE.data,
                s.du_spts.data[dim * per_dim ..][0..per_dim],
                s.du_fpts.data[dim * per_dim_f ..][0..per_dim_f],
                .overwrite,
            );
        }
    }

    // ---- Element <-> face coupling ----
    //
    // `geo.fpt2gfpt` / `fpt2gfpt_slot` map an element-local flux point to its
    // global flux point and side, so these are pure gather/scatter loops.

    /// Element flux-point solution -> the faces' two-sided state.
    pub fn scatterUToFaces(s: *Solver) void {
        const ele = &s.quad.ele;
        const mesh = s.mesh;

        for (0..s.n_eles) |e| {
            for (0..ele.n_fpts) |fpt| {
                const gf = mesh.fpt2gfpt.get(fpt, e);
                if (gf == geo_mod.none) continue;
                const slot = mesh.fpt2gfpt_slot.get(fpt, e);
                for (0..s.n_vars) |n| {
                    s.faces.u.at(slot, n, gf).* = s.u_fpts.get(fpt, n, e);
                }
                // The viscous flux reads u_ldg; on an interior face it is the
                // same state, and applyBcs overwrites the boundary side.
                if (s.config.equation.viscous) {
                    for (0..s.n_vars) |n| {
                        s.faces.u_ldg.at(slot, n, gf).* = s.u_fpts.get(fpt, n, e);
                    }
                }
            }
        }
    }

    /// Common normal flux -> each element's own flux-point array.
    pub fn gatherCommonFFromFaces(s: *Solver) void {
        const ele = &s.quad.ele;
        const mesh = s.mesh;

        for (0..s.n_eles) |e| {
            for (0..ele.n_fpts) |fpt| {
                const gf = mesh.fpt2gfpt.get(fpt, e);
                if (gf == geo_mod.none) {
                    // A collapsed face carries no flux
                    for (0..s.n_vars) |n| s.f_comm.at(fpt, n, e).* = 0.0;
                    continue;
                }
                const slot = mesh.fpt2gfpt_slot.get(fpt, e);
                for (0..s.n_vars) |n| {
                    s.f_comm.at(fpt, n, e).* = s.faces.f_comm.get(slot, n, gf);
                }
            }
        }
    }

    /// Common interface solution -> each element, for the gradient correction.
    pub fn gatherCommonUFromFaces(s: *Solver) void {
        const ele = &s.quad.ele;
        const mesh = s.mesh;

        for (0..s.n_eles) |e| {
            for (0..ele.n_fpts) |fpt| {
                const gf = mesh.fpt2gfpt.get(fpt, e);
                if (gf == geo_mod.none) {
                    for (0..s.n_vars) |n| s.u_comm.at(fpt, n, e).* = 0.0;
                    continue;
                }
                const slot = mesh.fpt2gfpt_slot.get(fpt, e);
                for (0..s.n_vars) |n| {
                    s.u_comm.at(fpt, n, e).* = s.faces.u_comm.get(slot, n, gf);
                }
            }
        }
    }

    /// Element flux-point gradients -> the faces' two-sided gradient.
    pub fn scatterGradToFaces(s: *Solver) void {
        const ele = &s.quad.ele;
        const mesh = s.mesh;

        for (0..s.n_eles) |e| {
            for (0..ele.n_fpts) |fpt| {
                const gf = mesh.fpt2gfpt.get(fpt, e);
                if (gf == geo_mod.none) continue;
                const slot = mesh.fpt2gfpt_slot.get(fpt, e);
                for (0..s.n_dims) |dim| {
                    for (0..s.n_vars) |n| {
                        s.faces.du.at(slot, dim, n, gf).* = s.du_fpts.get(dim, fpt, n, e);
                    }
                }
            }
        }
    }

    /// Domain period in direction `d`, from the mesh if it recorded one and
    /// otherwise from the extent of the mesh nodes.
    fn periodInDim(s: *const Solver, d: usize) f64 {
        const recorded = switch (d) {
            0 => s.mesh.periodic_dx,
            1 => s.mesh.periodic_dy,
            else => s.mesh.periodic_dz,
        };
        if (recorded > 0.0) return recorded;

        const b = s.meshBounds()[d];
        const lo = b[0];
        const hi = b[1];
        return hi - lo;
    }

    /// Hand the faces their geometry: the left element's outward unit normal,
    /// each side's face scaling, and the flux point coordinates.
    ///
    /// Called once from `init`. Only the left element writes `norm` and `coord`;
    /// both write their own `d_a`, since two cells can disagree about the
    /// reference-space size of a shared face.
    pub fn setFaceGeometry(s: *Solver) void {
        const ele = &s.quad.ele;
        const mesh = s.mesh;

        for (0..s.n_eles) |e| {
            for (0..ele.n_fpts) |fpt| {
                const gf = mesh.fpt2gfpt.get(fpt, e);
                if (gf == geo_mod.none) continue;
                const slot = mesh.fpt2gfpt_slot.get(fpt, e);

                s.faces.d_a.at(slot, gf).* = s.d_a_fpts.get(fpt, e);
                if (slot == 0) {
                    for (0..s.n_dims) |d| {
                        s.faces.norm.at(d, gf).* = s.norm_fpts.get(fpt, d, e);
                        s.faces.coord.at(d, gf).* = s.coord_fpts.get(fpt, d, e);
                    }
                }
            }
        }
    }

    /// Largest mismatch between the two sides' flux point coordinates, allowing
    /// for one periodic wrap.
    ///
    /// This is the one to use on a mesh with periodic boundaries: the two sides
    /// of a periodic interface sit a full domain length apart, which
    /// `fptPairingError` reports as a mismatch. A wrong pairing shows up as a
    /// distance that is neither zero nor a period.
    pub fn fptPairingErrorPeriodic(s: *const Solver) f64 {
        return s.pairingError(true);
    }

    /// Largest mismatch between the two sides' flux point coordinates.
    ///
    /// `geo.setupGlobalFpts` pairs the two sides of an interface by reversing
    /// the right element's traversal, which is exact for a conforming 2D mesh
    /// but is an *assumption*. This measures it: on a valid mesh the result is
    /// at roundoff, and anything larger means the pairing is wrong.
    pub fn fptPairingError(s: *const Solver) f64 {
        return s.pairingError(false);
    }

    fn pairingError(s: *const Solver, comptime wrap: bool) f64 {
        // Hoisted: periodInDim scans every mesh node when the mesh recorded no
        // period, which inside the flux point loop is quadratic in mesh size.
        var period: [3]f64 = .{ 0, 0, 0 };
        if (wrap) {
            for (0..s.n_dims) |d| period[d] = s.periodInDim(d);
        }

        const ele = &s.quad.ele;
        const mesh = s.mesh;

        // Left-side coordinates are already in faces.coord; compare the right.
        var worst: f64 = 0.0;
        for (0..s.n_eles) |e| {
            for (0..ele.n_fpts) |fpt| {
                const gf = mesh.fpt2gfpt.get(fpt, e);
                if (gf == geo_mod.none) continue;
                if (mesh.fpt2gfpt_slot.get(fpt, e) != 1) continue;

                var d2: f64 = 0.0;
                for (0..s.n_dims) |d| {
                    var diff = @abs(s.coord_fpts.get(fpt, d, e) - s.faces.coord.get(d, gf));
                    if (wrap) {
                        // Fold out a whole period: a periodic partner is exactly
                        // one domain length away, which is not an error.
                        if (period[d] > 0.0) diff = @min(diff, @abs(diff - period[d]));
                    }
                    d2 += diff * diff;
                }
                worst = @max(worst, @sqrt(d2));
            }
        }
        return worst;
    }

    // ---- Time stepping ----

    /// Advance one full time step with the configured RK scheme.
    pub fn update(s: *Solver) Error!void {
        const prev_time = s.flow_time;

        if (s.rk.n_stages > 1) @memcpy(s.u_ini.data, s.u_spts.data);

        // Intermediate stages: each advances from u_ini by alpha * dt
        const n_steps = if (s.rk.combines_stages) s.rk.n_stages - 1 else s.rk.n_stages;
        for (0..n_steps) |stage| {
            s.flow_time = prev_time + s.rk.c[stage] * s.dt;
            try s.computeResidual(stage);
            s.rkStage(stage);
        }

        if (s.rk.combines_stages) {
            const last = s.rk.n_stages - 1;
            s.flow_time = prev_time + s.rk.c[last] * s.dt;
            try s.computeResidual(last);
            s.rkCombine();
        }

        s.flow_time = prev_time + s.dt;
        s.current_iter += 1;
    }

    /// Intermediate RK stage: `u = u_ini - alpha dt / |J| divF`.
    ///
    /// The Jacobian determinant appears because `divF_spts` is a
    /// reference-space divergence; dividing by `|J|` returns it to physical
    /// space.
    pub fn rkStage(s: *Solver, stage: usize) void {
        const ele = &s.quad.ele;
        const src = if (s.rk.n_stages > 1) &s.u_ini else &s.u_spts;
        const a = s.rk.alpha[stage];

        for (0..ele.n_spts) |spt| {
            for (0..s.n_vars) |n| {
                for (0..s.n_eles) |e| {
                    const fac = a * s.dt / s.jaco_det_spts.get(spt, e);
                    s.u_spts.at(spt, n, e).* =
                        src.get(spt, n, e) - fac * s.divf_spts.get(stage, spt, n, e);
                }
            }
        }
    }

    /// Final RK combination: `u = u_ini - sum_stage beta dt / |J| divF_stage`.
    pub fn rkCombine(s: *Solver) void {
        const ele = &s.quad.ele;

        if (s.rk.n_stages > 1) @memcpy(s.u_spts.data, s.u_ini.data);

        for (0..s.rk.n_stages) |stage| {
            const b = s.rk.beta[stage];
            for (0..ele.n_spts) |spt| {
                for (0..s.n_vars) |n| {
                    for (0..s.n_eles) |e| {
                        const fac = b * s.dt / s.jaco_det_spts.get(spt, e);
                        s.u_spts.at(spt, n, e).* -= fac * s.divf_spts.get(stage, spt, n, e);
                    }
                }
            }
        }
    }

    // ---- Diagnostics ----

    /// L2 norm of the current residual, per variable.
    pub fn residualNorm(s: *Solver, stage: usize, out: []f64) void {
        const ele = &s.quad.ele;
        std.debug.assert(out.len >= s.n_vars);
        @memset(out[0..s.n_vars], 0.0);

        var total_vol: f64 = 0.0;
        for (s.vol) |v| total_vol += v;

        for (0..s.n_vars) |n| {
            var sum: f64 = 0.0;
            for (0..ele.n_spts) |spt| {
                for (0..s.n_eles) |e| {
                    const r = s.divf_spts.get(stage, spt, n, e);
                    // divF is already weighted by |J|, so the quadrature needs
                    // only the reference weights.
                    sum += ele.weights_spts[spt] * r * r / s.jaco_det_spts.get(spt, e);
                }
            }
            out[n] = @sqrt(sum / total_vol);
        }
    }
};

/// Accumulation mode for `gemm`.
const Mode = enum { overwrite, accumulate };

/// `C = A * B` (or `C += A * B`), all row-major and densely packed.
///
/// A is (m, k), B is (k, n), C is (m, n). This is the one kernel behind every
/// operator application: the solution arrays put the element index last, so
/// `n = n_vars * n_eles` and the inner loop runs over contiguous memory.
///
/// TODO: this is the solver's hot loop. ZEFR hands it to BLAS/GiMMiK; the
/// `spock` dependency already ships dgemm compute shaders for the GPU path.
fn gemm(
    m: usize,
    n: usize,
    k: usize,
    a: []const f64,
    b: []const f64,
    c: []f64,
    comptime mode: Mode,
) void {
    std.debug.assert(a.len >= m * k);
    std.debug.assert(b.len >= k * n);
    std.debug.assert(c.len >= m * n);

    for (0..m) |i| {
        const c_row = c[i * n ..][0..n];
        if (mode == .overwrite) @memset(c_row, 0.0);

        for (0..k) |kk| {
            const aik = a[i * k + kk];
            if (aik == 0.0) continue;
            const b_row = b[kk * n ..][0..n];
            for (c_row, b_row) |*cv, bv| cv.* += aik * bv;
        }
    }
}

const std = @import("std");

const cfg = @import("config.zig");
const flux = @import("flux.zig");
const geo_mod = @import("geo.zig");
const Geo = geo_mod.Geo;
const Element = @import("element.zig").Element;
const Quad = @import("eles/quads.zig").Quad;
const faces_mod = @import("faces.zig");
const testcase = @import("testcase.zig");
const gpu = @import("gpu.zig");
const Faces = faces_mod.Faces;

const Matrix = @import("util/matrix.zig").Matrix;
const Array3 = @import("util/array3.zig").Array3;
const Array4 = @import("util/array4.zig").Array4;

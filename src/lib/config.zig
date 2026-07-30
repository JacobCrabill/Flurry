const std = @import("std");
const Io = std.Io;
const ziggy = @import("ziggy");

// ── Enums ────────────────────────────────────────────────────

pub const Equation = enum { adv_diff, euler_ns };

pub const DtScheme = enum {
    // zig fmt: off
    euler, rk44, rk54, rkJ, steady, dirk34, esdirk43, esdirk64,
    // zig fmt: on
};

pub const FluxConvType = enum { rusanov };
pub const FluxViscType = enum { ldg };

pub const BoundaryCondition = enum {
    // zig fmt: off
    none, periodic, char, sup_in, sup_out, slip_wall,
    isothermal_noslip, isothermal_noslip_moving,
    adiabatic_noslip, adiabatic_noslip_moving,
    overset, symmetry, wall_closure, overset_closure,
    // zig fmt: on
};

pub const MotionType = enum { static, test1, test2, test3, circular_trans, rigid_body };
pub const IterativeMethod = enum { jac, mcgs };
pub const LinearSolver = enum { lu, inv, svd };
pub const MgCycle = enum { v, w };

// ── Config Structs ───────────────────────────────────────────

pub const CoreConfig = struct {
    n_dims: u8,
    mesh_file: []const u8,
    order: u8,
};

pub const EquationConfig = struct {
    equation: Equation,
    viscous: bool = false,
    disable_nondim: bool = false,
    source: bool = false,
    squeeze: bool = false,
    s_factor: f64 = 0.0,
    advdiff_A: [3]f64 = undefined,
    advdiff_D: f64 = undefined,
};

pub const TimeConfig = struct {
    dt_scheme: DtScheme,
    n_steps: u32,
    tfinal: f64 = 1e15,
    res_tol: f64 = 0.0,
    res_field: u32 = 0,
    dt: ?f64 = null,
    dt_type: u32 = 0,
    CFL: f64 = 1.0,
    CFL_type: u32 = 0,
    implicit_method: bool = false,
    implicit_steady: bool = false,
    adapt_dt: ?AdaptDtConfig = null,
    implicit: ?ImplicitConfig = null,
};

pub const AdaptDtConfig = struct {
    atol: f64 = 1e-5,
    rtol: f64 = 1e-5,
    pi_alpha: f64 = 0.7,
    pi_beta: f64 = 0.4,
    sfact: f64 = 0.8,
    maxfac: f64 = 2.5,
    minfac: f64 = 0.3,
    max_dt: f64 = 100.0,
};

pub const ImplicitConfig = struct {
    FDA_Jacobian: bool = false,
    linear_solver: LinearSolver = .lu,
    pseudo_time: ?PseudoTimeConfig = null,
};

pub const PseudoTimeConfig = struct {
    CFL_tau: f64 = 1.0,
};

pub const RestartConfig = struct {
    restart_file: []const u8 = "",
    restart_case: []const u8 = "",
    restart_type: u32 = 0,
    restart_iter: u32 = 0,
    restart_npart: i32 = -1,
};

pub const MultigridConfig = struct {
    mg_cycle: MgCycle = .v,
    FMG_vcycles: u32 = 1,
    p_multi: bool = false,
    rel_fac: f64 = 1.0,
    mg_levels: []u32,
    mg_steps: []u32,
};

pub const OutputConfig = struct {
    output_prefix: []const u8,
    write_paraview: bool = true,
    write_pyfr: bool = false,
    plot_surfaces: bool = false,
    plot_overset: bool = false,
    write_LHS: bool = false,
    write_RHS: bool = false,
    write_freq: u32,
    report_freq: u32,
    res_type: u32 = 0,
    force_freq: u32 = 0,
    error_freq: u32 = 0,
    turb_stat_freq: u32 = 0,
    write_tavg_freq: u32 = 0,
    tavg_freq: u32 = 100,
};

pub const TestCaseConfig = struct {
    test_case: u32 = 0,
    err_field: u32 = 0,
    n_qpts_1d: u32 = 5,
};

pub const FluxConfig = struct {
    fconv_type: FluxConvType = .rusanov,
    fvisc_type: FluxViscType = .ldg,
    rus_k: f64 = 0.0,
    ldg_b: f64 = 0.5,
    ldg_tau: f64 = 1.0,
    spt_type: []const u8 = "Legendre",
};

pub const GasPropertiesConfig = struct {
    T_gas: f64 = 291.15,
    gamma: f64 = 1.4,
    R: f64 = 286.9,
    prandtl: f64 = 0.72,
    S: f64 = 120.0,
};

pub const FreestreamConfig = struct {
    rho_fs: f64 = 1.4,
    P_fs: f64 = 1.0,
    mach_fs: f64 = 0.2,
    Re_fs: f64 = 200.0,
    L_fs: f64 = 1.0,
    T_fs: f64 = 300.0,
    norm_fs: [3]f64 = .{ 1.0, 0.0, 0.0 },
    fix_vis: bool = false,
};

pub const WallConditionsConfig = struct {
    mach_wall: f64 = 0.0,
    T_wall: f64 = 300.0,
    norm_wall: [3]f64 = .{ 1.0, 0.0, 0.0 },
};

pub const FilteringConfig = struct {
    filt_on: u32 = 0,
    sen_write: u32 = 1,
    sen_norm: u32 = 1,
    sen_Jfac: f64 = 1.0,
    alpha: f64 = 1.0,
    filtexp: f64 = 2.0,
    nonlin_exp: f64 = 2.0,
};

pub const OversetConfig = struct {
    overset_grids: [][]const u8,
    grid_types: []i32,
};

pub const MotionConfig = struct {
    motion_type: MotionType = .static,
    circular_trans: ?CircularTransConfig = null,
    rigid_body: ?RigidBodyConfig = null,
};

pub const CircularTransConfig = struct {
    move_Ax: f64,
    move_Ay: f64,
    move_Fx: f64,
    move_Fy: f64,
    move_Az: f64 = 0.0,
    move_Fz: f64 = 0.0,
};

pub const RigidBodyConfig = struct {
    g: f64 = 0.0,
    full_6dof: bool = false,
    v0: [3]f64 = .{ 0, 0, 0 },
    w0: [3]f64 = .{ 0, 0, 0 },
    mass: f64,
    Imat: [9]f64,
};

/// Map from mesh boundary name -> BC enum. Ziggy dictionaries (`{"name": val, ...}`)
/// deserialize directly into this, no hand-rolled entry-list + build step needed.
pub const BoundaryConditionsConfig = struct {
    mesh_bounds: ziggy.Dictionary(BoundaryCondition) = .empty,
};

/// Generate a uniform Cartesian mesh instead of reading one from a file
/// (`meshType = CREATE_MESH` in Flurry-cpp). Presence of this section is what
/// selects mesh creation, so `core.mesh_file` is ignored when it is set.
///
/// Note the axis each boundary names, which follows Flurry-cpp and is *not*
/// consistent between 2D and 3D:
///   - 2D: `bottom`/`top` are y = ymin/ymax, `left`/`right` are x = xmin/xmax
///   - 3D: `bottom`/`top` are z = zmin/zmax, `left`/`right` are x = xmin/xmax,
///         and `back`/`front` are y = ymin/ymax
pub const CreateMeshConfig = struct {
    nx: u32 = 10,
    ny: u32 = 10,
    /// Ignored when `core.n_dims` is 2
    nz: u32 = 10,

    xmin: f64 = -10.0,
    xmax: f64 = 10.0,
    ymin: f64 = -10.0,
    ymax: f64 = 10.0,
    zmin: f64 = -10.0,
    zmax: f64 = 10.0,

    bc_bottom: BoundaryCondition = .periodic,
    bc_right: BoundaryCondition = .periodic,
    bc_top: BoundaryCondition = .periodic,
    bc_left: BoundaryCondition = .periodic,
    /// 3D only
    bc_front: BoundaryCondition = .periodic,
    /// 3D only
    bc_back: BoundaryCondition = .periodic,
};

pub const SignalConfig = struct {
    catch_signals: bool = false,
};

/// Top-level parsed config.
pub const Config = struct {
    core: CoreConfig,
    equation: EquationConfig,
    time: TimeConfig,
    restart: ?RestartConfig = null,
    multigrid: MultigridConfig,
    output: OutputConfig,
    test_case: TestCaseConfig,
    flux: FluxConfig,
    gas_properties: GasPropertiesConfig,
    freestream: FreestreamConfig,
    wall_conditions: WallConditionsConfig,
    filtering: FilteringConfig,
    overset: ?OversetConfig = null,
    motion: ?MotionConfig = null,
    boundary_conditions: BoundaryConditionsConfig,
    signals: SignalConfig,
    create_mesh: ?CreateMeshConfig = null,
};

/// Arena-backed parse result. Ziggy has no recursive `free`, and (by default)
/// borrows some string data — notably `Dictionary` keys — directly from the
/// source buffer rather than copying it. So the arena owns both the source
/// text and everything deserialized from it, keeping their lifetimes tied
/// together. Call `deinit()` to release everything at once.
pub const ParsedConfig = struct {
    value: Config,
    arena_state: std.heap.ArenaAllocator,

    pub fn deinit(pc: *ParsedConfig) void {
        pc.arena_state.deinit();
    }
};

// ── Loader (parse + initialize) ──────────────────────────────

pub const loader = struct {
    /// Read file using std.Io API (same pattern as zigdown).
    fn readFile(io: Io, alloc: std.mem.Allocator, cwd: Io.Dir, path: []const u8) ![]u8 {
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path_len = try cwd.realPathFile(io, path, &path_buf);
        const realpath = path_buf[0..path_len];

        var file: Io.File = try cwd.openFile(io, realpath, .{});
        defer file.close(io);

        var read_buf: [4096]u8 = undefined;
        var fr = file.reader(io, &read_buf);
        return try fr.interface.allocRemaining(alloc, .unlimited);
    }

    /// Parse a .cfg.ziggy file. Returns an arena-backed `ParsedConfig` —
    /// caller must call `.deinit()` to free it. The source text is read
    /// straight into the arena so it stays alive as long as the parsed
    /// value (see `ParsedConfig` doc comment for why that matters).
    /// On a parse error, a human-readable diagnostic is written to stderr.
    pub fn parse(
        io: Io,
        alloc: std.mem.Allocator,
        cwd: Io.Dir,
        path: []const u8,
    ) !ParsedConfig {
        var arena_state: std.heap.ArenaAllocator = .init(alloc);
        errdefer arena_state.deinit();
        const arena = arena_state.allocator();

        const content = try readFile(io, arena, cwd, path);
        const src = try arena.allocSentinel(u8, content.len, 0);
        @memcpy(src, content);

        const value = try deserialize(io, arena, path, src);
        return .{ .value = value, .arena_state = arena_state };
    }

    /// Parse from a null-terminated string. Diagnostics (if any) go to stderr.
    /// `src` must outlive the returned `ParsedConfig` (see its doc comment).
    pub fn parseString(
        io: Io,
        alloc: std.mem.Allocator,
        src: [:0]const u8,
    ) !ParsedConfig {
        var arena_state: std.heap.ArenaAllocator = .init(alloc);
        errdefer arena_state.deinit();
        const value = try deserialize(io, arena_state.allocator(), null, src);
        return .{ .value = value, .arena_state = arena_state };
    }

    fn deserialize(
        io: Io,
        arena: std.mem.Allocator,
        path: ?[]const u8,
        src: [:0]const u8,
    ) !Config {
        var meta: ziggy.Deserializer.Meta = undefined;
        return ziggy.deserializeLeaky(Config, arena, src, &meta, .{}) catch |err| {
            if (err != error.OutOfMemory) reportError(io, arena, path, src, &meta, err);
            return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.ParseError,
            };
        };
    }

    /// Best-effort diagnostic print to stderr; parsing has already failed, so
    /// any error here is swallowed rather than shadowing the real failure.
    fn reportError(
        io: Io,
        alloc: std.mem.Allocator,
        path: ?[]const u8,
        src: [:0]const u8,
        meta: *ziggy.Deserializer.Meta,
        err: ziggy.Deserializer.Error,
    ) void {
        var stderr = Io.File.stderr();
        var buf: [4096]u8 = undefined;
        var w = stderr.writer(io, &buf);
        meta.reportErrors(alloc, .{}, path, src, err, &w.interface) catch return;
        w.interface.flush() catch return;
    }

    /// Post-parse init from CONFIG_PLAN.md "Init / Validation Logic".
    pub fn initialize(cfg: *Config) void {
        validateDimensions(cfg);
        applyTimeDerivations(cfg);
        applyMotionDefaults(cfg);
        applyFilteringAutoDisable(cfg);
    }

    fn validateDimensions(cfg: *Config) void {
        switch (cfg.core.n_dims) {
            2, 3 => {},
            else => unreachable, // would fail ziggy schema validation first
        }
    }

    fn applyTimeDerivations(cfg: *Config) void {
        const t = &cfg.time;
        if (t.dt_scheme == .steady) t.dt = null;

        switch (t.dt_scheme) {
            .steady, .dirk34, .esdirk43, .esdirk64 => t.implicit_method = true,
            else => {},
        }

        if (!t.implicit_method) t.implicit_steady = false;
    }

    fn applyMotionDefaults(cfg: *Config) void {
        const m = cfg.motion orelse return;
        // motion disabled when static
        if (m.motion_type == .static) cfg.motion = null;
    }

    fn applyFilteringAutoDisable(cfg: *Config) void {
        const f = &cfg.filtering;
        if (f.filt_on > 0 and cfg.core.order <= 1) f.filt_on = 0;
        if (cfg.output.error_freq == 0) cfg.test_case.n_qpts_1d = 0;
    }

    pub const Error = error{ OutOfMemory, FileNotFound, ParseError };
};

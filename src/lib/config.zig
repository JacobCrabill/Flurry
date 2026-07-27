/// The all-up simulation config struct
pub const Config = struct {
    physics: PhysicsConfig = .{},
    ic_type: InitialConditionType = .uniform,
    numerics: NumericsConfig = .{},
    iter_config: IterationConfig = .{},
    output_config: OutputConfig = .{},
    /// Restart from a checkpoint?
    restart: ?RestartConfig = null,
};

pub const PhysicsConfig = struct {
    equation: Equation = .navier_stokes,
    viscous: bool = false,
    /// Farfield boundary conditions.
    boundary_conditions: NavierStokesState,
    /// Initial conditions default to the farfield boundary conditions
    initial_conditions: ?NavierStokesState = null,
};

/// Navier-Stokes state specification.
/// Used for farfield boundary conditions and initial condition.
pub const NavierStokesState = struct {
    rho: f64,
    u: f64,
    v: f64,
    w: f64,
    p: f64,
};

pub const NumericsConfig = struct {
    /// Polynomial order of the solution
    order: u8 = 1,
    riemann_type: RiemannType = .rusanov,
    /// Time integration settings
    time_config: TimeIntegrationConfig = .{},
};

/// Settings related to iterations
pub const IterationConfig = struct {
    /// Max iteration to run until
    iter_max: usize = 0,
    /// Max simulation time (in simulated seconds) to run until
    max_time: f64 = 0,
};

pub const TimeIntegrationConfig = struct {
    /// Simulation time step (normalized seconds)
    dt: f64 = 0,
    /// Time integration method
    method: TimeIntegrationMethod = .rk45,
};

pub const TimeIntegrationMethod = enum(u8) {
    euler,
    rk2,
    rk44,
    rk45,
};

pub const OutputConfig = struct {
    /// Filename prefix for solution output
    output_file_name: []const u8 = "flurry-out",
    /// Period (in iterations) at which to output the solution
    output_period: usize = 1000,
    /// Output file format
    output_type: OutputType = .paraview,
};

/// File format for the solution output
pub const OutputType = enum(u8) {
    paraview,
    tecplot,
    pyfr,
};

/// Restart the solution from a checkpoint
pub const RestartConfig = struct {
    restart_iter: usize = 0,
    restart_file_name: []const u8,
};

/// Which equation is being solved.
/// (Note: inviscid vs. viscous is handled via the 'viscous' flag)
pub const Equation = enum(u8) {
    advection_diffusion,
    navier_stokes,
};

pub const InitialConditionType = enum(u8) {
    uniform,
    isentropic_vortex,
    guassian_bump,
};

pub const RiemannType = enum(u8) {
    rusanov,
    roe,
};

/// Parameters for Sutherland's Law
pub const SutherlandParams = struct {
    mu_gas: f64,
    t_gas: f64,
    s_gas: f64,
    rt_inf: f64,
    mu_inf: f64,
    c_sth: f64,
    fix_fis: bool,
};

pub const ViscousParams = struct {
    /// Penalty factor for the LDG viscous flux
    pen_fact: f64,
    /// Bias parameter for the LDG viscous flux
    tau: f64,
    /// Reynolds number
    reynolds: f64,
    /// Reference length for the Reynolds number
    rel_len: f64,
    /// Mach number
    mach: f64,
};

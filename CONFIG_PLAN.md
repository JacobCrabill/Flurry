# Config Plan — Zig Rewrite of ZEFR Input Parameters

## Overview

Hierarchical `Config` struct with optional sub-structs. Optionality driven by:

- Equation choice (EulerNS vs AdvDiff)
- Feature toggles (implicit_method, adapt_dt, motion, overset, restart, pseudo_time, etc.)
- Derived/computed values (not stored in config; computed during init)

## Enum / Union Candidates

| C++ type                            | Zig replacement                                              | Values                                                                                                                                                                                                                               |
| ----------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `equation` (string → enum)        | `Equation` enum                                              | `.adv_diff`, `.euler_ns`                                                                                                                                                                                                             |
| `dt_scheme` (string)                | `DtScheme` enum                                              | `.euler`, `.rk44`, `.rk54`, `.rkJ`, `.steady`, `.dirk34`, `.esdirk43`, `.esdirk64`                                                                                                                                                   |
| `fconv_type` (string)               | `FluxConvType` enum                                          | `.rusanov` (only one; keep extensible)                                                                                                                                                                                               |
| `fvisc_type` (string)               | `FluxViscType` enum                                          | `.ldg`                                                                                                                                                                                                                               |
| Boundary conditions (`bcStr2Num`)   | `BoundaryCondition` enum                                     | `.none`, `.periodic`, `.char`, `.sup_in`, `.sup_out`, `.slip_wall`, `.isothermal_noslip`, `.isothermal_noslip_moving`, `.adiabatic_noslip`, `.adiabatic_noslip_moving`, `.overset`, `.symmetry`, `.wall_closure`, `.overset_closure` |
| `motion_type` (int)                 | `MotionType` enum                                            | `.static`, `.test1`, `.test2`, `.test3`, `.circular_trans`, `.rigid_body`                                                                                                                                                            |
| `iterative_method` (string → int) | `IterativeMethod` enum                                       | `.jac`, `.mcgs`                                                                                                                                                                                                                      |
| `linear_solver` (string → int)    | `LinearSolver` enum                                          | `.lu`, `.inv`, `.svd`                                                                                                                                                                                                                |
| `mg_cycle` (string)                 | `MgCycle` enum                                               | `.v`, `.w` (standard V/W cycle)                                                                                                                                                                                                      |
| `dt_type` (int)                     | `DtType` enum or keep int if values are arbitrary thresholds | TBD — depends on what 0/1/etc. mean in the solver code                                                                                                                                                                             |
| `CFL_type` (int)                    | Same as dt_type                                              | TBD                                                                                                                                                                                                                                  |
| `dtau_type` / `CFL_tau_type`        | Same family                                                  | TBD                                                                                                                                                                                                                                  |

## Struct Hierarchy

```
Config
├── core: CoreConfig              // always present
├── equation: EquationConfig      // depends on equation choice
├── time: TimeConfig              // always present (dt_scheme-specific fields live inside)
│   └── adapt_dt: ?AdaptDtConfig  // optional — only if adapt_dt = true
│   └── implicit: ?ImplicitConfig // optional — only if implicit_method = true
│       └── pseudo_time: ?PseudoTimeConfig  // optional — only if pseudo_time = true
├── restart: ?RestartConfig       // optional — only if restart = true
├── multigrid: MultigridConfig    // always present (defaults are no-op)
├── output: OutputConfig          // always present
├── test_case: TestCaseConfig     // always present
├── flux: FluxConfig              // always present
├── gas_properties: GasPropertiesConfig  // always present
├── freestream: FreestreamConfig        // always present (EulerNS only; AdvDiff ignores)
├── wall_conditions: WallConditionsConfig       // always present
├── filtering: FilteringConfig          // always present
├── overset: ?OversetConfig             // optional — only if overset = true
├── motion: ?MotionConfig               // optional — only if motion = true and motion_type != .static
│   └── circular_trans: ?CircularTransConfig  // optional sub-struct for CIRCULAR_TRANS params
│   └── rigid_body: ?RigidBodyConfig    // optional sub-struct for RIGID_BODY params
├── boundary_conditions: BoundaryConditionsConfig // always present
└── signals: SignalConfig               // always present (single bool)
```

## Detailed Struct Definitions

### CoreConfig

```
n_dims: u8                    // 2 or 3
mesh_file: []const u8         // required, no default
order: u8                     // polynomial order
```

### EquationConfig

```
equation: Equation            // .adv_diff | .euler_ns
viscous: bool                 // only meaningful for EulerNS; false if AdvDiff
disable_nondim: bool          // default false
source: bool                  // default false
squeeze: bool                 // default false
s_factor: f64                 // default 0.0 (only used if squeeze)

// AdvDiff-specific (set when equation == .adv_diff)
advdiff_A: [3]f64             // convection coefficient vector, default [1.0, 1.0, 0.0] for 2D? actually 3 components always
advdiff_D: f64                // diffusion coefficient, default 0.1 (or 0 if !viscous)
```

### TimeConfig

```
dt_scheme: DtScheme           // required — no default, must be specified
n_steps: u32                  // required, no default
tfinal: f64                   // default 1e15
res_tol: f64                  // default 0.0
res_field: u32                // default 0

// dt field — only read if scheme is not "Steady"
dt: ?f64                      // null for steady-state; value otherwise

dt_type: u32                  // default 0 (TBD → enum)
CFL: f64                      // default 1.0 (only used when dt_type != 0)
CFL_type: u32                 // default 0 (TBD → enum)

n_stages: u8                  // derived from dt_scheme; computed at init, not stored in input

// Implicit flag — derived from dt_scheme but also user-settable
implicit_method: bool         // true for steady/dirk/esdirk schemes; can be overridden
implicit_steady: bool         // only meaningful when implicit_method = true
```

### AdaptDtConfig (optional)

```
atol: f64          // default 1e-5
rtol: f64          // default 1e-5
pi_alpha: f64      // default 0.7
pi_beta: f64       // default 0.4
sfact: f64         // default 0.8
maxfac: f64        // default 2.5
minfac: f64        // default 0.3
max_dt: f64        // default 100.0
```

### RestartConfig (optional)

```
restart_file: []const u8     // default ""
restart_case: []const u8     // default ""
restart_type: u32            // default 0
restart_iter: u32            // default 0
restart_npart: i32           // default -1
```

### MultigridConfig

```
mg_cycle: MgCycle       // default .v
FMG_vcycles: u32        // default 1
p_multi: bool           // default false
rel_fac: f64            // default 1.0
mg_levels: []u32        // required
mg_steps: []u32         // required
```

### OutputConfig

```
output_prefix: []const u8     // required, no default
write_paraview: bool          // short in C++ — use bool; default true
write_pyfr: bool              // default false
plot_surfaces: bool           // default false
plot_overset: bool            // default false
write_LHS: bool               // default false
write_RHS: bool               // default false
write_freq: u32               // required, no default
report_freq: u32              // required, no default

res_type: u32                 // TBD — what values? (could be enum)

force_freq: u32       // default 0
error_freq: u32       // default 0
turb_stat_freq: u32   // default 0

// Time-averaging — derived relationship in C++:
// tavg is true iff write_tavg_freq > 0 AND tavg_freq > 0
write_tavg_freq: u32  // default 0
tavg_freq: u32        // default 100 (only meaningful if write_tavg_freq > 0)
```

### TestCaseConfig

```
test_case: u32      // default 0
err_field: u32      // default 0
n_qpts_1d: u32      // default 5; auto-set to 0 when error_freq == 0
```

### FluxConfig

```
fconv_type: FluxConvType   // default .rusanov
fvisc_type: FluxViscType   // default .ldg
rus_k: f64                 // default 0.0 (Rusanov penalty parameter)
ldg_b: f64                 // default 0.5 (LDG stabilization parameter)
ldg_tau: f64               // default 1.0 (LDG penalty parameter)
spt_type: []const u8       // polynomial type, e.g. "Legendre" — keep as string for extensibility → could be enum later
```

### GasPropertiesConfig

```
T_gas: f64      // default 291.15 (Sutherland temperature)
gamma: f64      // default 1.4
R: f64          // default 286.9 (specific gas constant)
prandtl: f64    // default 0.72
S: f64          // default 120.0 (Sutherland constant)
```

### FreestreamConfig

```
rho_fs: f64       // default 1.4
P_fs: f64         // default 1.0
mach_fs: f64      // default 0.2
Re_fs: f64        // default 200.0
L_fs: f64         // default 1.0
T_fs: f64         // default 300.0
norm_fs: [3]f64   // freestream normal vector, default [1.0, 0.0, 0.0]
fix_vis: bool     // default false (fixed viscosity vs Sutherland)

// Derived at init (not stored in config):
// v_mag_fs — computed from mach * sqrt(gamma * P / rho)
// V_fs[3]    — velocity vector = norm_fs * v_mag_fs
// T_tot_fs   — total temperature
// P_tot_fs   — total pressure
```

### WallConditionsConfig

```
mach_wall: f64      // default 0.0
T_wall: f64         // default 300.0
norm_wall: [3]f64   // wall normal vector, default [1.0, 0.0, 0.0]

// Derived at init (not stored in config):
// V_wall[3] — wall velocity vector
```

### FilteringConfig

```
filt_on: u32        // default 0; auto-disabled if order <= 1
sen_write: u32      // default 1
sen_norm: u32       // default 1
sen_Jfac: f64       // default 1.0
alpha: f64          // default 1.0
filtexp: f64        // default 2.0
nonlin_exp: f64     // default 2.0

// Derived/unused in current code:
// filt_maxLevels, shockcapture, limiter, filt2on, filt_gamma, filtexp2, alpha2 — not read from input.cpp
//   (these may come from elsewhere or be hardcoded)
```

### OversetConfig (optional)

```
overset_grids: [][]const u8    // list of mesh files per overset grid
grid_types: []i32              // 0=background, 1=geometry; defaults to [1] for all if undersized
```

### MotionConfig (optional)

```
motion_type: MotionType   // default .static — but motion = false when .static
```

### CircularTransConfig (optional, nested inside MotionConfig)

```
move_Ax: f64      // amplitude x
move_Ay: f64      // amplitude y
move_Fx: f64      // frequency x
move_Fy: f64      // frequency y
move_Az: ?f64     // amplitude z (only in 3D, default 0.0)
move_Fz: ?f64     // frequency z (only in 3D, default 0.0)
```

### RigidBodyConfig (optional, nested inside MotionConfig)

```
g: f64            // gravity, default 0.0
full_6dof: bool   // default false — fully dynamic vs prescribed motion
v0: [3]f64        // initial translational velocity, default [0,0,0]
w0: [3]f64        // initial angular velocity, default [0,0,0]
mass: f64         // required, no default (body mass)
Imat: [9]f64      // inertia tensor (symmetric; Ixy=Iyx etc. auto-applied), diagonal required

// Derived from diagonal: off-diagonal Ixz/Izx, Iyz/Izy set automatically
```

### BoundaryConditionsConfig

```
mesh_bounds: std.StringHashMap(BoundaryCondition)  // map from mesh boundary name → BC enum
```

### SignalConfig

```
catch_signals: bool   // default false
```

## Init / Validation Logic (post-parse, pre-simulation)

These transformations happen after parsing but before simulation start, mirroring the C++
`initialize_inputs()` and post-parse logic:

1. **nStages** — set from dt_scheme enum value
2. **implicit_method / implicit_steady** — derived from dt_scheme
3. **dt field** — null if dt_scheme == .steady
4. **CFL / CFL_tau** — only read when dt_type/CFL/dtau_type != 0
5. **AdaptDt sub-struct** — only present when adapt_dt = true
6. **Restart sub-struct** — only present when restart = true
7. **Motion sub-struct** — only present when motion = true and motion_type != .static
8. **CircularTrans / RigidBody** — only present for matching motion_type
9. **PseudoTime** — only present inside ImplicitConfig when pseudo_time = true
10. **nQpts1D** — auto-set to 0 when error_freq == 0
11. **tavg** — derived from write_tavg_freq > 0 AND tavg_freq > 0
12. **Freestream normalization** — V_fs, v_mag_fs, T_tot_fs, P_tot_fs computed from mach_fs +
    norm_fs + gas properties
13. **Nondimensionalization** (if viscous and EulerNS) — mu, rho_ref, P_ref, etc. all computed in
    `apply_nondim()`
14. **Filtering auto-disable** — if filt_on && order <= 1, disable filtering with warning
15. **Report freq defaults** — report_BMconv_freq = iterBM_max when NM freq > 0 and BM freq == 0

## Serialization Format Recommendation

Use a simple key-value format (similar to existing) or consider Zig-compatible serialization:

- **TOML**: human-readable, supports sections natively matching the struct hierarchy
- **JSON**: widely supported, easy to validate with zig-json
- **Custom INI-style**: match current input file format exactly for zero migration cost

For TOML, the mapping is nearly 1:1:

```toml
[core]
n_dims = 2
mesh_file = "cylinder.mesh"
order = 3

[equation]
type = "EulerNS"
viscous = true
disable_nondim = false

[time]
dt_scheme = "RK44"
n_steps = 1000
tfinal = 1e15

[time.implicit]
FDA_Jacobian = false
linear_solver = "LU"
```

## Parameters NOT in Config (computed/dynamic)

These are derived during init or runtime — not stored as input:

- `v_mag_fs`, `V_fs[3]` — computed from freestream conditions
- `T_tot_fs`, `P_tot_fs` — total conditions from isentropic relations
- `mu`, `rho_ref`, `P_ref`, `T_ref`, `R_ref`, `c_sth`, `rt` — nondimensionalization constants
- `V_wall[3]` — wall velocity vector
- `nStages` — derived from dt_scheme
- `tavg` boolean — derived from write_tavg_freq + tavg_freq
- Implicit-derived Newton iteration max and Jacobian freeze flag

## Parameters Read Elsewhere (not in input.cpp)

These fields exist on InputStruct but are NOT read by `read_input_file()`:

| Field                                                                        | Likely source                          | Action                                     |
| ---------------------------------------------------------------------------- | -------------------------------------- | ------------------------------------------ |
| rank, nRanks, grank                                                          | MPI init                               | Not config — pass from runtime           |
| iter, time                                                                   | Runtime counters                       | Not config                                 |
| filt_maxLevels, shockcapture, limiter, filt2on, filt_gamma, filtexp2, alpha2 | Hardcoded or other files               | Keep as hardcoded defaults or separate cfg |
| u_fs, v_fs (scalar)                                                          | Part of V_fs vector                    | Remove — use V_fs[3] only                |
| u_wall, v_wall (scalar)                                                      | Part of V_wall vector                  | Remove — use V_wall[3] only              |
| rot_axis, rot_angle, xc, dxc, vc, dvc                                        | Motion-related, possibly set elsewhere | Add to MotionConfig if needed              |
| nGrids, gridType, gridID                                                     | Runtime mesh state                     | Not config                                 |

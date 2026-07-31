//! Tests for config.zig — Ziggy-based input parsing and post-parse init.
//!
//! Covers: successful parse of the cylinder sample, field-level value checks,
//! post-parse derivations, error paths (missing required field, invalid enum tag).

const std = @import("std");
const testing = std.testing;
const Io = std.Io;
const config = @import("config.zig");

const Loader = config.Loader;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse a null-terminated string. The test allocator is used so arena
/// allocations are tracked and freed automatically by `std.testing`.
fn parseTest(io: Io, src: [:0]const u8) !config.ParsedConfig {
    return Loader.parseString(io, std.testing.allocator, src);
}

// ---------------------------------------------------------------------------
// 1. Successful parse of the cylinder sample
// ---------------------------------------------------------------------------

test "parse cylinder.cfg.ziggy from samples/" {
    const io = std.testing.io;
    const alloc = std.testing.allocator;
    const cwd = Io.Dir.cwd();
    var pc = Loader.parse(io, alloc, cwd, "samples/cylinder.cfg.ziggy") catch |err| {
        return err;
    };
    defer pc.deinit();

    // Core fields
    try testing.expectEqual(@as(u8, 2), pc.value.core.n_dims);
    try testing.expect(std.mem.eql(u8, "cylinder_2d.mesh", pc.value.core.mesh_file));
    try testing.expectEqual(@as(u8, 3), pc.value.core.order);

    // Equation
    try testing.expectEqual(config.Equation.euler_ns, pc.value.equation.equation);
    try testing.expect(pc.value.equation.viscous);
    try testing.expect(!pc.value.equation.disable_nondim);

    // Time
    try testing.expectEqual(config.DtScheme.rk44, pc.value.time.dt_scheme);
    try testing.expectEqual(@as(u32, 1000), pc.value.time.n_steps);
    try expectApprox(1e-6, pc.value.time.dt.?, 1e-15);

    // Output
    try testing.expect(std.mem.eql(u8, "cylinder_rk3", pc.value.output.output_prefix));
    try testing.expectEqual(@as(u32, 100), pc.value.output.write_freq);

    // Flux
    try testing.expectEqual(config.FluxConvType.rusanov, pc.value.flux.fconv_type);
    try testing.expectEqual(config.FluxViscType.ldg, pc.value.flux.fvisc_type);

    // Gas properties
    try expectApprox(291.15, pc.value.gas_properties.T_gas, 1e-10);
    try expectApprox(1.4, pc.value.gas_properties.gamma, 1e-10);

    // Freestream
    try expectApprox(1.4, pc.value.freestream.rho_fs, 1e-10);
    try expectApprox(0.2, pc.value.freestream.mach_fs, 1e-10);

    // Fixed-size arrays go down a different path in the deserializer than
    // slices do; check the elements, not just that parsing succeeded.
    try testing.expectEqual([3]f64{ 1.0, 0.0, 0.0 }, pc.value.freestream.norm_fs);

    // Wall conditions
    try expectApprox(300.0, pc.value.wall_conditions.T_wall, 1e-10);
    try testing.expectEqual([3]f64{ 1.0, 0.0, 0.0 }, pc.value.wall_conditions.norm_wall);

    // Boundary conditions — dictionary iteration
    var bc_count: usize = 0;
    var it = pc.value.boundary_conditions.mesh_bounds.fields.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        const bc = entry.value_ptr.*;
        _ = name;
        _ = bc;
        bc_count += 1;
    }
    try testing.expectEqual(@as(usize, 4), bc_count);

    // Signals
    try testing.expect(!pc.value.signals.catch_signals);
}

test "a section whose fields all have defaults can be left out" {
    // Only `core`, `equation`, `time` and `output` have nothing sensible to
    // fall back on. Everything else should be omittable, so a small case file
    // stays small instead of spelling out values it does not care about.
    const src: [:0]const u8 =
        \\.core = .{ .n_dims = 2, .mesh_file = "m.msh", .order = 3 },
        \\.equation = .{ .equation = .euler_ns },
        \\.time = .{ .dt_scheme = .rk44, .n_steps = 10, .dt = 1e-3 },
        \\.output = .{ .output_prefix = "o", .write_freq = 0, .report_freq = 0 },
        \\
    ;
    var pc = try parseTest(std.testing.io, src);
    defer pc.deinit();

    try testing.expectEqual(@as(f64, 1.4), pc.value.gas_properties.gamma);
    try testing.expectEqual(config.FluxConvType.rusanov, pc.value.flux.fconv_type);
    try testing.expectEqual(@as(u32, 0), pc.value.boundary_conditions.mesh_bounds.fields.count());
    try testing.expect(pc.value.create_mesh == null);
    try testing.expect(pc.value.restart == null);
}

test "an empty struct literal takes the section's defaults" {
    // `.{}` says "this section, all defaults" explicitly. Upstream ziggy
    // rejected it outright; see the VENDOR FIX in Deserializer.zig.
    const src: [:0]const u8 =
        \\.core = .{ .n_dims = 2, .mesh_file = "m.msh", .order = 3 },
        \\.equation = .{ .equation = .euler_ns },
        \\.time = .{ .dt_scheme = .rk44, .n_steps = 10, .dt = 1e-3 },
        \\.output = .{ .output_prefix = "o", .write_freq = 0, .report_freq = 0 },
        \\.gas_properties = .{},
        \\.freestream = .{ .mach_fs = 0.5 },
        \\
    ;
    var pc = try parseTest(std.testing.io, src);
    defer pc.deinit();

    try testing.expectEqual(@as(f64, 1.4), pc.value.gas_properties.gamma);
    try testing.expectEqual(@as(f64, 286.9), pc.value.gas_properties.R);
    // ...and a partly-filled section still keeps the defaults it did not name
    try testing.expectEqual(@as(f64, 0.5), pc.value.freestream.mach_fs);
    try testing.expectEqual(@as(f64, 1.4), pc.value.freestream.rho_fs);
}

// ---------------------------------------------------------------------------
// 2. Post-parse initialization derivations
// ---------------------------------------------------------------------------

test "steady scheme → dt = null" {
    const src: [:0]const u8 = @embedFile("testdata/config/steady.cfg.ziggy");
    var pc = try parseTest(std.testing.io, src);
    defer pc.deinit();

    try Loader.initialize(&pc.value);

    try testing.expect(pc.value.time.dt == null);
}

test "rk44 scheme → dt present" {
    const src: [:0]const u8 = @embedFile("testdata/config/rk44.cfg.ziggy");
    var pc = try parseTest(std.testing.io, src);
    defer pc.deinit();

    try Loader.initialize(&pc.value);

    try testing.expect(pc.value.time.dt != null);
}

test "n_qpts_1d auto-set to 0 when error_freq == 0" {
    const src: [:0]const u8 = @embedFile("testdata/config/no_error_freq.cfg.ziggy");
    var pc = try parseTest(std.testing.io, src);
    defer pc.deinit();

    try Loader.initialize(&pc.value);

    try testing.expectEqual(@as(u32, 0), pc.value.test_case.n_qpts_1d);
}

// ---------------------------------------------------------------------------
// 3. Error paths
// ---------------------------------------------------------------------------

test "missing required field produces ParseError" {
    // core.mesh_file is required (no default).
    const src: [:0]const u8 = @embedFile("testdata/config/missing_mesh_file.cfg.ziggy");
    errdefer {} // suppress unused warning

    const result = parseTest(std.testing.io, src) catch |err| {
        if (err == error.ParseError or err == error.OutOfMemory) return;
        std.debug.print("unexpected error: {}\n", .{err});
        return err;
    };
    // If we reach here without an error, fail the test.
    _ = result;
    return error.ExpectedParseError;
}

test "invalid enum tag produces ParseError" {
    const src: [:0]const u8 = @embedFile("testdata/config/invalid_enum.cfg.ziggy");
    errdefer {} // suppress unused warning

    const result = parseTest(std.testing.io, src) catch |err| {
        if (err == error.ParseError or err == error.OutOfMemory) return;
        std.debug.print("unexpected error: {}\n", .{err});
        return err;
    };
    // If we reach here without an error, fail the test.
    _ = result;
    return error.ExpectedParseError;
}

test "create_mesh section is optional and round-trips" {
    // Absent in the standard fixtures: mesh creation is off unless asked for.
    var rk = try parseTest(std.testing.io, @embedFile("testdata/config/rk44.cfg.ziggy"));
    defer rk.deinit();
    try testing.expectEqual(@as(?config.CreateMeshConfig, null), rk.value.create_mesh);

    var pc = try parseTest(std.testing.io, @embedFile("testdata/config/create_mesh.cfg.ziggy"));
    defer pc.deinit();

    const cm = pc.value.create_mesh orelse return error.ExpectedCreateMeshSection;
    try testing.expectEqual(@as(u32, 4), cm.nx);
    try testing.expectEqual(@as(u32, 8), cm.ny);
    try testing.expectEqual(@as(u32, 2), cm.nz);
    try expectApprox(-1.0, cm.xmin, 1e-12);
    try expectApprox(4.0, cm.ymax, 1e-12);
    try expectApprox(-0.5, cm.zmin, 1e-12);
    try testing.expectEqual(config.BoundaryCondition.slip_wall, cm.bc_bottom);
    try testing.expectEqual(config.BoundaryCondition.symmetry, cm.bc_top);
    try testing.expectEqual(config.BoundaryCondition.sup_in, cm.bc_left);
    try testing.expectEqual(config.BoundaryCondition.sup_out, cm.bc_right);
    try testing.expectEqual(config.BoundaryCondition.characteristic, cm.bc_front);
    try testing.expectEqual(config.BoundaryCondition.periodic, cm.bc_back);
}

// ---------------------------------------------------------------------------
// 4. Boundary condition dictionary iteration
// ---------------------------------------------------------------------------

test "boundary conditions iterate in insertion order" {
    const io = std.testing.io;
    const alloc = std.testing.allocator;
    const cwd = Io.Dir.cwd();
    var pc = Loader.parse(io, alloc, cwd, "samples/cylinder.cfg.ziggy") catch |err| {
        return err;
    };
    defer pc.deinit();

    // Expected order from cylinder.cfg.ziggy: inlet, outlet, cylinder, farfield.
    var names: [4][]const u8 = undefined;
    var i: usize = 0;
    var it = pc.value.boundary_conditions.mesh_bounds.fields.iterator();
    while (it.next()) |entry| {
        names[i] = entry.key_ptr.*;
        i += 1;
    }
    try testing.expectEqual(@as(usize, 4), i);
    try testing.expect(std.mem.eql(u8, "inlet", names[0]));
    try testing.expect(std.mem.eql(u8, "outlet", names[1]));
    try testing.expect(std.mem.eql(u8, "cylinder", names[2]));
    try testing.expect(std.mem.eql(u8, "farfield", names[3]));
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn expectApprox(a: f64, b: f64, tol: f64) !void {
    if (std.math.isNan(a) and std.math.isNan(b)) return;
    if (@abs(a - b) > tol * @max(1.0, @max(@abs(a), @abs(b)))) {
        std.debug.print("expected {d}, got {d} (tol {d})\n", .{ b, a, tol });
        return error.TestExpectedApproxEq;
    }
}

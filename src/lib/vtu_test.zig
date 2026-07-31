//! Tests for vtu.zig. These check the bytes that land in the file, not just
//! that the writer ran: a `.vtu` whose declared offsets do not match its
//! appended blob still "writes fine" and then fails to open.

const std = @import("std");
const testing = std.testing;
const Io = std.Io;

const cfg = @import("config.zig");
const driver = @import("driver.zig");
const vtu = @import("vtu.zig");

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn testConfig(order: u8, nx: u32, ny: u32) cfg.Config {
    var config: cfg.Config = undefined;
    config.core = .{ .n_dims = 2, .mesh_file = "", .order = order };
    config.equation = .{
        .equation = .euler_ns,
        .viscous = false,
        .advdiff_A = .{ 1.0, 0.5, 0.0 },
        .advdiff_D = 0.1,
    };
    config.time = .{ .dt_scheme = .rk44, .n_steps = 0, .dt = 1e-3 };
    config.restart = null;
    config.output = .{ .output_prefix = "test", .write_freq = 0, .report_freq = 0 };
    config.test_case = .{ .test_case = .uniform, .err_field = 0, .n_qpts_1d = 0 };
    config.flux = .{};
    config.gas_properties = .{};
    config.freestream = .{ .mach_fs = 0.3, .norm_fs = .{ 1.0, 0.0, 0.0 }, .fix_vis = true };
    config.wall_conditions = .{};
    config.boundary_conditions = .{};
    config.signals = .{};
    config.create_mesh = .{
        .nx = nx,
        .ny = ny,
        .xmin = 0.0,
        .xmax = 2.0,
        .ymin = 0.0,
        .ymax = 2.0,
        .bc_bottom = .characteristic,
        .bc_top = .characteristic,
        .bc_left = .characteristic,
        .bc_right = .characteristic,
    };
    return config;
}

/// A written document, plus where its raw blob begins.
const Doc = struct {
    bytes: []u8,
    blob: usize,

    fn deinit(d: Doc, gpa: std.mem.Allocator) void {
        gpa.free(d.bytes);
    }

    /// The `u32` byte count introducing the array declared at `offset`.
    fn count(d: Doc, offset: usize) u32 {
        return std.mem.readInt(u32, d.bytes[d.blob + offset ..][0..4], .little);
    }

    fn f64At(d: Doc, offset: usize, i: usize) f64 {
        const at = d.blob + offset + 4 + i * 8;
        return @bitCast(std.mem.readInt(u64, d.bytes[at..][0..8], .little));
    }

    fn f32At(d: Doc, offset: usize, i: usize) f32 {
        const at = d.blob + offset + 4 + i * 4;
        return @bitCast(std.mem.readInt(u32, d.bytes[at..][0..4], .little));
    }

    fn u32At(d: Doc, offset: usize, i: usize) u32 {
        const at = d.blob + offset + 4 + i * 4;
        return std.mem.readInt(u32, d.bytes[at..][0..4], .little);
    }

    fn u8At(d: Doc, offset: usize, i: usize) u8 {
        return d.bytes[d.blob + offset + 4 + i];
    }

    /// The `offset="N"` of the DataArray whose declaration contains `needle`.
    fn declaredOffset(d: Doc, needle: []const u8) !usize {
        const header = d.bytes[0..d.blob];
        const at = std.mem.indexOf(u8, header, needle) orelse return error.NotFound;
        const tail = header[at..];
        const key = std.mem.indexOf(u8, tail, "offset=\"") orelse return error.NotFound;
        const rest = tail[key + 8 ..];
        const end = std.mem.indexOfScalar(u8, rest, '"') orelse return error.NotFound;
        return std.fmt.parseInt(usize, rest[0..end], 10);
    }

    fn headerValue(d: Doc, comptime T: type, needle: []const u8) !T {
        const header = d.bytes[0..d.blob];
        const at = std.mem.indexOf(u8, header, needle) orelse return error.NotFound;
        const rest = header[at + needle.len ..];
        var end: usize = 0;
        while (end < rest.len and (std.ascii.isDigit(rest[end]))) end += 1;
        return std.fmt.parseInt(T, rest[0..end], 10);
    }
};

fn writeDoc(gpa: std.mem.Allocator, run: *driver.Run) !Doc {
    var out: Io.Writer.Allocating = .init(gpa);
    errdefer out.deinit();
    try vtu.write(&run.solver, gpa, &out.writer);

    const bytes = try out.toOwnedSlice();
    errdefer gpa.free(bytes);

    const tag = std.mem.indexOf(u8, bytes, "<AppendedData") orelse return error.NotFound;
    const underscore = std.mem.indexOfScalarPos(u8, bytes, tag, '_') orelse return error.NotFound;
    return .{ .bytes = bytes, .blob = underscore + 1 };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "the appended blob is exactly where the XML says it is" {
    const gpa = testing.allocator;

    // A `.vtu` whose declared offsets disagree with its blob is silently
    // well-formed XML and opens as garbage, so walk the two against each other.
    const config = testConfig(3, 3, 2);

    var run: driver.Run = undefined;
    try run.init(gpa, testing.io, &config, .{});
    defer run.deinit();

    const doc = try writeDoc(gpa, &run);
    defer doc.deinit(gpa);

    const n_eles: usize = 6;
    const n_ppts: usize = 16; // (order+1)^2
    const n_sub: usize = 9; // order^2
    const n_points = n_eles * n_ppts;
    const n_cells = n_eles * n_sub;

    try testing.expectEqual(n_points, try doc.headerValue(usize, "NumberOfPoints=\""));
    try testing.expectEqual(n_cells, try doc.headerValue(usize, "NumberOfCells=\""));

    // Each array's introducing count must match its declared type and length
    const expect = [_]struct { needle: []const u8, bytes: usize }{
        .{ .needle = "Name=\"rho\"", .bytes = n_points * 8 },
        .{ .needle = "Name=\"xmom\"", .bytes = n_points * 8 },
        .{ .needle = "Name=\"ymom\"", .bytes = n_points * 8 },
        .{ .needle = "Name=\"energy\"", .bytes = n_points * 8 },
        .{ .needle = "<Points>", .bytes = n_points * 3 * 4 },
        .{ .needle = "Name=\"connectivity\"", .bytes = n_cells * 4 * 4 },
        .{ .needle = "Name=\"offsets\"", .bytes = n_cells * 4 },
        .{ .needle = "Name=\"types\"", .bytes = n_cells },
    };

    var running: usize = 0;
    for (expect) |e| {
        const off = try doc.declaredOffset(e.needle);
        try testing.expectEqual(running, off);
        try testing.expectEqual(@as(u32, @intCast(e.bytes)), doc.count(off));
        running += 4 + e.bytes;
    }

    // ...and the blob holds all of it and nothing more but the closing tags
    try testing.expect(doc.blob + running <= doc.bytes.len);
    try testing.expect(std.mem.endsWith(u8, doc.bytes, "</AppendedData>\n</VTKFile>\n"));
}

test "the sub-cells tile every element exactly once" {
    const gpa = testing.allocator;

    // The tessellation is what ParaView actually draws. If a sub-quad's corners
    // are listed out of order it comes out bow-tied with negative area, and if
    // the per-element base index is wrong it reaches into a neighbour's points.
    const config = testConfig(3, 4, 3);

    var run: driver.Run = undefined;
    try run.init(gpa, testing.io, &config, .{});
    defer run.deinit();

    const doc = try writeDoc(gpa, &run);
    defer doc.deinit(gpa);

    const n_eles: usize = 12;
    const n_ppts: usize = 16;
    const n_sub: usize = 9;
    const n_cells = n_eles * n_sub;

    const conn = try doc.declaredOffset("Name=\"connectivity\"");
    const offs = try doc.declaredOffset("Name=\"offsets\"");
    const types = try doc.declaredOffset("Name=\"types\"");
    const pts = try doc.declaredOffset("<Points>");

    var total_area: f64 = 0.0;
    for (0..n_cells) |c| {
        // A cell may only use points belonging to the element it came from
        const ele = c / n_sub;
        var corner: [4][2]f64 = undefined;
        for (0..4) |k| {
            const p = doc.u32At(conn, 4 * c + k);
            try testing.expect(p >= ele * n_ppts);
            try testing.expect(p < (ele + 1) * n_ppts);
            corner[k] = .{ doc.f32At(pts, 3 * p), doc.f32At(pts, 3 * p + 1) };
            // 2D, so every point sits in the z = 0 plane
            try testing.expectEqual(@as(f32, 0.0), doc.f32At(pts, 3 * p + 2));
        }

        // Shoelace: positive means counter-clockwise, which is what VTK_QUAD
        // wants; a bow-tie or a reversed listing would not be.
        var area: f64 = 0.0;
        for (0..4) |k| {
            const a = corner[k];
            const b = corner[(k + 1) % 4];
            area += a[0] * b[1] - b[0] * a[1];
        }
        area *= 0.5;
        try testing.expect(area > 0.0);
        total_area += area;

        try testing.expectEqual(@as(u32, @intCast(4 * (c + 1))), doc.u32At(offs, c));
        try testing.expectEqual(@as(u8, 9), doc.u8At(types, c)); // VTK_QUAD
    }

    // The sub-cells cover the domain once over: no gaps, no overlaps. The
    // mesh is [0,2]^2, and f32 coordinates set the tolerance.
    try testing.expectApproxEqAbs(@as(f64, 4.0), total_area, 1e-5);
}

test "a field the basis represents exactly reaches the plot points intact" {
    const gpa = testing.allocator;

    // The plot points are not the solution points, so the values written are
    // an interpolation. A polynomial of degree <= order is in the span of the
    // basis, so it must come back exactly -- anything else means `oppE_ppts`
    // or the point ordering is wrong.
    const config = testConfig(3, 4, 3);

    var run: driver.Run = undefined;
    try run.init(gpa, testing.io, &config, .{});
    defer run.deinit();

    const s = &run.solver;
    const ele = &s.quad.ele;

    const field = struct {
        fn at(x: f64, y: f64, n: usize) f64 {
            const c: f64 = @floatFromInt(n + 1);
            return c * (0.3 + 0.7 * x - 0.4 * y + 0.2 * x * y + 0.1 * x * x);
        }
    };

    for (0..ele.n_spts) |spt| {
        for (0..s.n_eles) |e| {
            const x = s.coord_spts.get(spt, 0, e);
            const y = s.coord_spts.get(spt, 1, e);
            for (0..s.n_vars) |n| s.u_spts.at(spt, n, e).* = field.at(x, y, n);
        }
    }

    const doc = try writeDoc(gpa, &run);
    defer doc.deinit(gpa);

    const pts = try doc.declaredOffset("<Points>");
    const names = [_][]const u8{ "Name=\"rho\"", "Name=\"xmom\"", "Name=\"ymom\"", "Name=\"energy\"" };

    for (names, 0..) |name, n| {
        const off = try doc.declaredOffset(name);
        for (0..s.n_eles * ele.n_ppts) |p| {
            // Coordinates are written single precision, so compare against the
            // field evaluated at the coordinates the file actually carries.
            const x: f64 = doc.f32At(pts, 3 * p);
            const y: f64 = doc.f32At(pts, 3 * p + 1);
            try testing.expectApproxEqAbs(field.at(x, y, n), doc.f64At(off, p), 1e-6);
        }
    }
}

test "plot points span each element, corner to corner" {
    const gpa = testing.allocator;

    // Order 1 puts a plot point at each corner and nowhere else, so the written
    // coordinates must be the cell corners themselves.
    const config = testConfig(1, 2, 2);

    var run: driver.Run = undefined;
    try run.init(gpa, testing.io, &config, .{});
    defer run.deinit();

    const doc = try writeDoc(gpa, &run);
    defer doc.deinit(gpa);

    const pts = try doc.declaredOffset("<Points>");

    // Cell 0 of a 2x2 mesh on [0,2]^2 is the unit square at the origin, and
    // plot points run x-fastest.
    const want = [4][2]f32{ .{ 0, 0 }, .{ 1, 0 }, .{ 0, 1 }, .{ 1, 1 } };
    for (want, 0..) |w, p| {
        try testing.expectApproxEqAbs(w[0], doc.f32At(pts, 3 * p), 1e-6);
        try testing.expectApproxEqAbs(w[1], doc.f32At(pts, 3 * p + 1), 1e-6);
    }

    // One sub-cell per element at order 1
    try testing.expectEqual(@as(usize, 4), try doc.headerValue(usize, "NumberOfCells=\""));
    try testing.expectEqual(@as(usize, 16), try doc.headerValue(usize, "NumberOfPoints=\""));
}

test "order 0 is rejected rather than crashing" {
    const gpa = testing.allocator;

    // Order 0 has one point per cell and so no sub-cell to draw. It never gets
    // as far as the writer: the element's own point sets reject it first, which
    // they used to do by panicking on an unsigned reverse range.
    const config = testConfig(0, 2, 2);

    var run: driver.Run = undefined;
    try testing.expectError(error.UnsupportedOrder, run.init(gpa, testing.io, &config, .{}));
}

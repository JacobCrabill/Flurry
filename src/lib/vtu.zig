//! ParaView output: one binary `.vtu` per solution write.
//!
//! Ported from ZEFR's `FRSolver::write_solution`. A high-order cell cannot be
//! drawn directly, so each one is written as a patch of linear sub-cells over
//! its equispaced plot points -- `(order)^2` sub-quads sharing `(order+1)^2`
//! points. Neighbouring cells do *not* share points: each cell contributes its
//! own copy, which is what makes a discontinuous solution look discontinuous.
//!
//! The file is VTK's XML "UnstructuredGrid" with `AppendedData encoding="raw"`:
//! the XML names each array and gives its byte offset into one raw blob at the
//! end, and every array in the blob is preceded by a `u32` byte count.
//!
//! Not ported: the `.pvtu` index for MPI runs, overset blanking, grid velocity
//! for moving meshes, the filter sensor field, and the PyFR writer.

pub const Error = error{
    /// Order 0 has no sub-cells to draw; there is nothing to plot but a point
    UnsupportedOrder,
} || std.mem.Allocator.Error;

/// VTK cell types: a bilinear quadrilateral and a trilinear hexahedron.
const vtk_quad: u8 = 9;
const vtk_hexahedron: u8 = 12;

/// VTK always wants three components per point, whatever the mesh dimension.
const vtk_dims = 3;

const endian: std.builtin.Endian = .little;

/// Write `<prefix>/<prefix>_<iter>.vtu` under `dir`, creating the directory if
/// it is not there yet.
///
/// Returns the path written, borrowed from `path_buf`.
pub fn writeSolution(
    s: *const Solver,
    gpa: std.mem.Allocator,
    io: Io,
    dir: Io.Dir,
    prefix: []const u8,
    path_buf: *[std.fs.max_path_bytes]u8,
) ![]const u8 {
    const ele = s.element();
    if (ele.n_spts_1d < 2) return error.UnsupportedOrder;

    // ZEFR puts every write for a case in a directory named after the case, so
    // a long run does not bury the working directory in files.
    dir.createDirPath(io, prefix) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };

    const path = try std.fmt.bufPrint(
        path_buf,
        "{s}/{s}_{d:0>9}.vtu",
        .{ prefix, prefix, s.current_iter },
    );

    var file = try dir.createFile(io, path, .{});
    defer file.close(io);

    var buf: [64 * 1024]u8 = undefined;
    var fw = file.writer(io, &buf);
    try write(s, gpa, &fw.interface);
    try fw.interface.flush();

    return path;
}

/// Emit the whole document to `w`.
pub fn write(s: *const Solver, gpa: std.mem.Allocator, w: *Io.Writer) !void {
    const ele = s.element();
    if (ele.n_spts_1d < 2) return error.UnsupportedOrder;

    const n_ppts = ele.n_ppts;
    const n_sub_1d = ele.n_spts_1d - 1;

    // A cell is drawn as a patch of linear sub-cells over its plot points:
    // squares in 2D, cubes in 3D.
    const n_sub = std.math.pow(usize, n_sub_1d, s.n_dims);
    const verts_per_cell: usize = if (s.n_dims == 3) 8 else 4;
    const cell_type: u8 = if (s.n_dims == 3) vtk_hexahedron else vtk_quad;

    const n_points = s.n_eles * n_ppts;
    const n_cells = s.n_eles * n_sub;

    // The solution and the plot point coordinates, both (ppt, *, ele)
    var u_ppts = try Array3(f64).init(gpa, n_ppts, s.n_vars, s.n_eles);
    defer u_ppts.deinit(gpa);
    var coord_ppts = try Array3(f64).init(gpa, n_ppts, s.n_dims, s.n_eles);
    defer coord_ppts.deinit(gpa);

    try s.extrapolateToPpts(&u_ppts);
    try s.plotPointCoords(gpa, &coord_ppts);

    try w.writeAll(
        \\<?xml version="1.0"?>
        \\<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
        \\
    );

    // ZEFR's metadata comments, which its restart path reads back
    try w.print("<!-- ORDER {d} -->\n", .{ele.order});
    try w.print("<!-- TIME {e:.16} -->\n", .{s.flow_time});
    try w.print("<!-- ITER {d} -->\n", .{s.current_iter});

    try w.print(
        "<UnstructuredGrid>\n<Piece NumberOfPoints=\"{d}\" NumberOfCells=\"{d}\">\n",
        .{ n_points, n_cells },
    );

    // Every array in the appended blob is `[u32 byte count][payload]`, and the
    // offset quoted here counts both.
    var offset: usize = 0;

    try w.writeAll("<PointData>\n");
    for (0..s.n_vars) |n| {
        try w.print(
            "<DataArray type=\"Float64\" Name=\"{s}\" format=\"appended\" offset=\"{d}\"/>\n",
            .{ varName(s.config.equation.equation, s.n_dims, n), offset },
        );
        offset += @sizeOf(u32) + n_points * @sizeOf(f64);
    }
    try w.writeAll("</PointData>\n");

    try w.print(
        "<Points>\n<DataArray type=\"Float32\" NumberOfComponents=\"3\" " ++
            "format=\"appended\" offset=\"{d}\"/>\n</Points>\n",
        .{offset},
    );
    offset += @sizeOf(u32) + n_points * vtk_dims * @sizeOf(f32);

    try w.writeAll("<Cells>\n");
    try w.print(
        "<DataArray type=\"UInt32\" Name=\"connectivity\" format=\"appended\" offset=\"{d}\"/>\n",
        .{offset},
    );
    offset += @sizeOf(u32) + n_cells * verts_per_cell * @sizeOf(u32);

    try w.print(
        "<DataArray type=\"UInt32\" Name=\"offsets\" format=\"appended\" offset=\"{d}\"/>\n",
        .{offset},
    );
    offset += @sizeOf(u32) + n_cells * @sizeOf(u32);

    try w.print(
        "<DataArray type=\"UInt8\" Name=\"types\" format=\"appended\" offset=\"{d}\"/>\n",
        .{offset},
    );
    try w.writeAll("</Cells>\n</Piece>\n</UnstructuredGrid>\n");

    // ---- the raw blob ----
    try w.writeAll("<AppendedData encoding=\"raw\">\n_");

    for (0..s.n_vars) |n| {
        try writeCount(w, n_points * @sizeOf(f64));
        for (0..s.n_eles) |e| {
            for (0..n_ppts) |ppt| try writeF64(w, u_ppts.get(ppt, n, e));
        }
    }

    try writeCount(w, n_points * vtk_dims * @sizeOf(f32));
    for (0..s.n_eles) |e| {
        for (0..n_ppts) |ppt| {
            for (0..vtk_dims) |d| {
                // A 2D mesh still needs a z: ParaView reads it as a flat slab.
                const x = if (d < s.n_dims) coord_ppts.get(ppt, d, e) else 0.0;
                try writeF32(w, @floatCast(x));
            }
        }
    }

    // Sub-cell corners in each cell's own point block. Plot points run x
    // fastest, then y, then z, so a step in the next index up is a factor of
    // n_spts_1d further along -- which is what `corner` below walks.
    const n = ele.n_spts_1d;
    try writeCount(w, n_cells * verts_per_cell * @sizeOf(u32));
    for (0..s.n_eles) |e| {
        const base = e * n_ppts;
        // A 2D mesh has one layer, so the k loop runs once and contributes
        // nothing to the index.
        const n_layers = if (s.n_dims == 3) n_sub_1d else 1;
        for (0..n_layers) |k| {
            for (0..n_sub_1d) |j| {
                for (0..n_sub_1d) |i| {
                    const corner = struct {
                        fn at(nn: usize, a: usize, b: usize, c: usize) usize {
                            return a + nn * (b + nn * c);
                        }
                    }.at;

                    // VTK's own corner order: the low face counter-clockwise,
                    // then the high one, which is also Gmsh's hex ordering.
                    const lo = [4][2]usize{ .{ 0, 0 }, .{ 1, 0 }, .{ 1, 1 }, .{ 0, 1 } };
                    for (0..verts_per_cell) |v| {
                        const d = lo[v % 4];
                        const dk: usize = if (v < 4) 0 else 1;
                        const p = corner(n, i + d[0], j + d[1], k + dk);
                        try w.writeInt(u32, @intCast(base + p), endian);
                    }
                }
            }
        }
    }

    // Running end-of-cell index into the connectivity array
    try writeCount(w, n_cells * @sizeOf(u32));
    for (1..n_cells + 1) |c| try w.writeInt(u32, @intCast(verts_per_cell * c), endian);

    try writeCount(w, n_cells * @sizeOf(u8));
    for (0..n_cells) |_| try w.writeByte(cell_type);

    try w.writeAll("\n</AppendedData>\n</VTKFile>\n");
}

/// The `u32` byte count that introduces every appended array.
fn writeCount(w: *Io.Writer, n_bytes: usize) !void {
    try w.writeInt(u32, @intCast(n_bytes), endian);
}

fn writeF64(w: *Io.Writer, x: f64) !void {
    try w.writeInt(u64, @bitCast(x), endian);
}

fn writeF32(w: *Io.Writer, x: f32) !void {
    try w.writeInt(u32, @bitCast(x), endian);
}

/// Name ParaView shows for conserved variable `n`. ZEFR's spelling, so its
/// filters and state files carry over.
fn varName(equation: cfg.Equation, n_dims: usize, n: usize) []const u8 {
    return switch (equation) {
        .adv_diff => "u",
        .euler_ns => switch (n) {
            0 => "rho",
            1 => "xmom",
            2 => "ymom",
            3 => if (n_dims == 3) "zmom" else "energy",
            else => "energy",
        },
    };
}

const std = @import("std");
const Io = std.Io;

const cfg = @import("config.zig");
const Solver = @import("solver.zig").Solver;
const Array3 = @import("util/array3.zig").Array3;

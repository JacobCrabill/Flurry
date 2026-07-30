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

/// VTK cell type for a bilinear quadrilateral.
const vtk_quad: u8 = 9;

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
    const ele = &s.quad.ele;
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
    const ele = &s.quad.ele;
    if (ele.n_spts_1d < 2) return error.UnsupportedOrder;

    const n_ppts = ele.n_ppts;
    const n_sub_1d = ele.n_spts_1d - 1;
    const n_sub = n_sub_1d * n_sub_1d;

    const n_points = s.n_eles * n_ppts;
    const n_cells = s.n_eles * n_sub;

    // The solution and the plot point coordinates, both (ppt, *, ele)
    var u_ppts = try Array3(f64).init(gpa, n_ppts, s.n_vars, s.n_eles);
    defer u_ppts.deinit(gpa);
    var coord_ppts = try Array3(f64).init(gpa, n_ppts, s.n_dims, s.n_eles);
    defer coord_ppts.deinit(gpa);

    s.extrapolateToPpts(&u_ppts);
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
    offset += @sizeOf(u32) + n_cells * 4 * @sizeOf(u32);

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

    // Sub-cell corners, counter-clockwise, in each cell's own point block
    try writeCount(w, n_cells * 4 * @sizeOf(u32));
    for (0..s.n_eles) |e| {
        const base = e * n_ppts;
        for (0..n_sub_1d) |i| {
            for (0..n_sub_1d) |j| {
                // Plot points run x-fastest, so a row step is n_spts_1d
                const lo = i * ele.n_spts_1d + j;
                const hi = lo + ele.n_spts_1d;
                for ([_]usize{ lo, lo + 1, hi + 1, hi }) |p| {
                    try w.writeInt(u32, @intCast(base + p), endian);
                }
            }
        }
    }

    // Running end-of-cell index into the connectivity array
    try writeCount(w, n_cells * @sizeOf(u32));
    for (1..n_cells + 1) |c| try w.writeInt(u32, @intCast(4 * c), endian);

    try writeCount(w, n_cells * @sizeOf(u8));
    for (0..n_cells) |_| try w.writeByte(vtk_quad);

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

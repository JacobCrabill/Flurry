/// Geometry loading and processing
pub const Geo = struct {
    gpa: std.mem.Allocator,
    io: std.Io,

    config: cfg.Config,

    n_dims: usize = 2,
    n_eles: usize = 0,
    n_verts: usize = 0,
    n_edges: usize = 0,
    n_faces: usize = 0,
    n_int_faces: usize = 0,
    n_bnd_faces: usize = 0,
    n_mpi_faces: usize = 0,

    /// Number of boundaries
    n_bounds: usize = 0,

    // MESH_TYPE mesh_type;
    n_nodes_per_cell: usize = 0,

    // Basic [essential] Connectivity Data
    c2v: Matrix(usize) = .empty,

    /// Physical position of vertices
    xv: Matrix(f64) = .empty,

    // ---- Connectivity Data ----
    c2e: Matrix(usize) = .empty,
    c2b: Matrix(usize) = .empty,
    e2c: Matrix(usize) = .empty,
    e2v: Matrix(usize) = .empty,
    v2e: Matrix(usize) = .empty,
    v2v: Matrix(usize) = .empty,
    v2c: Matrix(usize) = .empty,
    c2f: Matrix(usize) = .empty,
    f2v: Matrix(usize) = .empty,
    f2c: Matrix(usize) = .empty,
    c2c: Matrix(usize) = .empty,
    c2ac: Matrix(usize) = .empty,

    v2nv: std.ArrayList(usize) = .empty,
    v2nc: std.ArrayList(usize) = .empty,
    c2nv: std.ArrayList(usize) = .empty,
    c2nf: std.ArrayList(usize) = .empty,
    f2nv: std.ArrayList(usize) = .empty,
    ctype: std.ArrayList(usize) = .empty,

    int_faces: std.ArrayList(usize) = .empty,
    bnd_faces: std.ArrayList(usize) = .empty,
    mpi_faces: std.ArrayList(usize) = .empty,
    mpi_cells: std.ArrayList(usize) = .empty,

    /// List of boundary conditions for each boundary
    /// TODO: use enum
    bc_list: std.ArrayList(usize) = .empty,

    /// List of boundaries given in mesh file
    bc_names: std.ArrayList([]const u8) = .empty,

    /// Boundary condition for each boundary face
    bc_type: std.ArrayList(usize) = .empty,

    /// List of node IDs on each boundary
    bnd_pts: Matrix(usize) = .empty,

    /// List of node IDs on each Gmsh boundary ("PhysicalName")
    bnd_pts_gmsh: std.ArrayList(std.ArrayList(usize)) = .empty,

    bc_id: std.ArrayList(usize) = .empty,
    n_gmsh_bnds: usize,

    /// Number of points on each boudary
    n_bnd_pts: std.ArrayList(usize) = .empty,

    /// List of nodes on each face (edge) for each boundary condition
    bc_faces: std.ArrayList(Matrix(usize)) = .empty,

    /// List of # of faces on each boundary
    n_faces_per_bnd: std.ArrayList(usize) = .empty,
    /// What processor lies to the 'right' of this face
    proc_r: std.ArrayList(usize) = .empty,
    /// The local mpiFace ID of each mpiFace on the opposite processor
    face_id_r: std.ArrayList(usize) = .empty,
    /// The global cell ID of the right cell on the opposite processor
    g_ic_r: std.ArrayList(usize) = .empty,
    /// Element-local face ID of MPI Face in left cell
    mpi_loc_f: std.ArrayList(usize) = .empty,
    /// Element-local face ID of MPI Face in right cell
    mpi_loc_f_r: std.ArrayList(usize) = .empty,
    /// Flag for whether an MPI face is also a periodic face
    mpi_periodic: std.ArrayList(usize) = .empty,

    // /// Type for each face: hole, internal, boundary, MPI, overset [-1,0,1,2,3]
    // face_type: std.ArrayList(FACE_TYPE) = .empty,

    pub fn readGmsh(geo: *Geo, file: []const u8) !void {
        const gpa = geo.gpa;
        const arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();

        const mesh_file: []const u8 = try std.Io.Dir.cwd().readFileAlloc(arena, file, arena.allocator(), .{});

        // var reader = std.Io.Reader.fixed(mesh_file);
        // _ = reader; // autofix

        // if (grid_rank == 0)
        //   std::cout << "Geo: Reading mesh file " << file_name << std::endl;

        // --- Read Boundary Conditions & Fluid Field(s) ---

        // Move cursor to $PhysicalNames
        var iter = std.mem.tokenizeScalar(u8, mesh_file, '\n');
        while (iter.next()) |line| {
            if (std.mem.find(u8, line, "$PhysicalNames")) |_| {
                std.debug.print("Found '$PhysicalNames'\n", .{});
                break;
            }
        }
        if (iter.peek() == null) {
            @panic("$PhysicalNames tag not found in Gmsh file!");
        }

        // Read the boundary count
        var line = iter.next().?;
        const n_bounds = try std.fmt.parseInt(usize, line, 10);
        std.debug.print("{d} physical names\n", .{n_bounds});

        // Parse each boundary
        var bc_names: std.ArrayList([]const u8) = .empty;
        try bc_names.resize(arena.allocator(), n_bounds);
        for (0..n_bounds) |_| {
            line = iter.next().?;
            // split by spaces
            var split = std.mem.tokenizeAny(u8, line, " \t");
            const ndim = try std.fmt.parseInt(usize, split.next().?, 10);
            const bcid = try std.fmt.parseInt(usize, split.next().?, 10);
            const name_quoted = split.next().?;
            const name_unquoted = try std.mem.tokenizeScalar(u8, name_quoted, '"').next().?;
            const name_lower = try std.ascii.allocLowerString(gpa, name_unquoted);
            try bc_names.append(arena.allocator(), name_lower);
            std.debug.print("{d} {d} {s}\n", .{ ndim, bcid, name_lower });
        }

        // CLAUDE TODO: Fill in the blanks
    }

    pub fn createMesh(geo: *Geo) !void {
        _ = geo; // autofix
    }
};

const std = @import("std");

const Matrix = @import("util/matrix.zig").Matrix;
const Array3 = @import("util/array3.zig").Array3;
const Array4 = @import("util/array4.zig").Array4;
const cfg = @import("config.zig");
const points = @import("points.zig");
const poly = @import("math/polynomials.zig");

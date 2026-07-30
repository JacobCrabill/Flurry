//! Geometry loading and connectivity processing

/// Cell (element) topology types supported by the solver.
///
/// Note that "collapsed" elements (tets, prisms) are stored as `hex` with
/// duplicated vertices, matching the original Flurry-cpp behavior.
pub const CellType = enum { tri, quad, hex };

/// Classification of each face. The numeric values are Flurry-cpp's, and the
/// sign is meaningful: `c2b` marks a face as "on a boundary" when the value is
/// greater than zero.
pub const FaceType = enum(i8) {
    internal = 0,
    boundary = 1,
    mpi = 2,
};

/// Stands in for the -1 that Flurry-cpp stored in its signed connectivity
/// matrices: "no such cell / face / boundary". The connectivity members are
/// unsigned here, so an out-of-range sentinel takes its place.
pub const none: usize = std.math.maxInt(usize);

/// Errors specific to deriving mesh connectivity.
pub const ConnError = error{
    /// A face is shared by more than two cells
    NonManifoldFace,
    /// `ctype` and `c2nf` disagree about how many faces a cell has
    InconsistentCellType,
    /// A cell type that does not exist in this many dimensions
    UnsupportedCellType,
    /// A face on a periodic boundary found no partner across the domain
    UnmatchedPeriodicFace,
    /// Periodic vertex merging aliased two distinct faces onto one, which
    /// happens when a periodic direction spans only two cells
    PeriodicDirectionTooThin,
    /// `core.n_dims` is neither 2 nor 3
    UnsupportedDimension,
    /// `setupGlobalFpts` was called before `processConnectivity`
    ConnectivityNotProcessed,
};

/// Errors specific to reading/parsing a Gmsh `.msh` file.
pub const GmshError = error{
    /// A required `$Section` tag was not found in the file
    MissingSection,
    /// The `$MeshFormat` version is neither 2.x nor 4.x
    UnsupportedMeshFormat,
    /// A line had fewer fields than the format requires
    MalformedMeshFile,
    /// A Gmsh PhysicalName has no matching entry in `mesh_bounds`
    UnrecognizedMeshBoundary,
    /// The mesh contains no physical group identifying the fluid region
    NoFluidRegion,
    /// The fluid region contains no elements
    NoInteriorCells,
    /// A Gmsh element type ID we don't know how to read
    UnrecognizedElementType,
    /// An element referenced a node tag that was never defined in `$Nodes`
    UnknownNodeTag,
};

/// Errors specific to generating a Cartesian mesh.
pub const CreateMeshError = error{
    /// `createMesh` was called without a `create_mesh` config section
    MissingCreateMeshConfig,
    /// `core.n_dims` is neither 2 nor 3
    UnsupportedDimension,
};

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
    ctype: std.ArrayList(CellType) = .empty,

    int_faces: std.ArrayList(usize) = .empty,
    bnd_faces: std.ArrayList(usize) = .empty,
    mpi_faces: std.ArrayList(usize) = .empty,
    mpi_cells: std.ArrayList(usize) = .empty,

    /// Boundary condition applied to each boundary, parallel to `bc_names`
    bc_list: std.ArrayList(cfg.BoundaryCondition) = .empty,

    /// List of boundaries given in mesh file (gpa-owned copies of the
    /// lower-cased Gmsh PhysicalNames), parallel to `bc_list`
    bc_names: std.ArrayList([]const u8) = .empty,

    /// Boundary condition for each boundary face, indexed by position in
    /// `bnd_faces`. `.none` marks a boundary face that matched no boundary.
    bc_type: std.ArrayList(cfg.BoundaryCondition) = .empty,

    /// List of node IDs on each boundary
    bnd_pts: Matrix(usize) = .empty,

    /// List of node IDs on each Gmsh boundary ("PhysicalName")
    bnd_pts_gmsh: std.ArrayList(std.ArrayList(usize)) = .empty,

    /// Boundary index (into `bc_list`) for each boundary face, indexed by
    /// position in `bnd_faces`. `none` where no boundary matched.
    bc_id: std.ArrayList(usize) = .empty,
    n_gmsh_bnds: usize = 0,

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

    // ---- Periodic box extents ----
    // Set by `createMesh`; used to match up periodic boundary faces.
    periodic_dx: f64 = 0.0,
    periodic_dy: f64 = 0.0,
    periodic_dz: f64 = 0.0,

    /// Type for each face: internal, boundary, MPI
    face_type: std.ArrayList(FaceType) = .empty,

    // ---- Global flux points ----
    // Built by `setupGlobalFpts`. Every face carries `n_fpts_per_face` *global*
    // flux points (gfpts), the points at which the two sides of an interface
    // exchange data. Interior gfpts come first; boundary ones occupy the tail,
    // so a boundary gfpt is `gfpt - n_gfpts_int` into `gfpt2bnd`.

    /// Flux points along a single face
    n_fpts_per_face: usize = 0,
    /// Flux points on a whole element, `n_faces * n_fpts_per_face`
    n_fpts_per_ele: usize = 0,

    n_gfpts: usize = 0,
    n_gfpts_int: usize = 0,
    n_gfpts_bnd: usize = 0,

    /// Global flux point of each element-local flux point, (fpt, ele).
    /// `none` where a cell face is collapsed.
    fpt2gfpt: Matrix(usize) = .empty,

    /// Which side of its interface an element-local flux point is on:
    /// 0 = left, 1 = right. (fpt, ele)
    fpt2gfpt_slot: Matrix(u8) = .empty,

    /// Boundary index (into `bc_list`) of each boundary gfpt, indexed by
    /// `gfpt - n_gfpts_int`. `none` if the face matched no boundary.
    gfpt2bnd: std.ArrayList(usize) = .empty,

    /// Face ID owning each global flux point
    gfpt2face: std.ArrayList(usize) = .empty,

    /// Release everything `readGmsh`, `createMesh` or `processConnectivity`
    /// allocated from `gpa`.
    pub fn deinit(geo: *Geo) void {
        const gpa = geo.gpa;

        // Mesh
        geo.c2v.deinit(gpa);
        geo.xv.deinit(gpa);
        geo.bnd_pts.deinit(gpa);
        geo.c2nv.deinit(gpa);
        geo.c2nf.deinit(gpa);
        geo.ctype.deinit(gpa);
        geo.n_bnd_pts.deinit(gpa);
        geo.n_faces_per_bnd.deinit(gpa);
        geo.bc_list.deinit(gpa);

        for (geo.bc_names.items) |name| gpa.free(name);
        geo.bc_names.deinit(gpa);

        // Connectivity
        geo.f2v.deinit(gpa);
        geo.e2v.deinit(gpa);
        geo.c2f.deinit(gpa);
        geo.c2b.deinit(gpa);
        geo.c2c.deinit(gpa);
        geo.f2c.deinit(gpa);
        geo.f2nv.deinit(gpa);
        geo.face_type.deinit(gpa);
        geo.int_faces.deinit(gpa);
        geo.bnd_faces.deinit(gpa);
        geo.bc_type.deinit(gpa);
        geo.bc_id.deinit(gpa);

        for (geo.bc_faces.items) |*mat| mat.deinit(gpa);
        geo.bc_faces.deinit(gpa);

        geo.fpt2gfpt.deinit(gpa);
        geo.fpt2gfpt_slot.deinit(gpa);
        geo.gfpt2bnd.deinit(gpa);
        geo.gfpt2face.deinit(gpa);
    }

    /// Read a Gmsh `.msh` file (ASCII format 2.x or 4.x) and populate the
    /// vertex positions (`xv`), cell-to-vertex connectivity (`c2v`, `c2nv`,
    /// `c2nf`, `ctype`), and boundary node lists (`bnd_pts`, `n_bnd_pts`).
    ///
    /// Physical groups are matched against `config.boundary_conditions.mesh_bounds`
    /// by lower-cased name; the group named "fluid" identifies the interior
    /// region and sets `n_dims`.
    pub fn readGmsh(geo: *Geo, file: []const u8) !void {
        var arena_state: std.heap.ArenaAllocator = .init(geo.gpa);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        report("Geo: Reading mesh file {s}\n", .{file});

        const src = try std.Io.Dir.cwd().readFileAlloc(geo.io, file, arena, .unlimited);
        return geo.parseGmsh(arena, src);
    }

    /// The body of `readGmsh`, operating on an already-loaded mesh file.
    ///
    /// `arena` is used for scratch data that lives only until this returns;
    /// everything stored on `geo` is allocated from `geo.gpa`.
    pub fn parseGmsh(geo: *Geo, arena: std.mem.Allocator, src: []const u8) !void {
        var scan: Scanner = .{ .src = src };

        const version = try readMeshFormat(&scan);

        // --- Read Boundary Conditions & Fluid Field(s) ---
        var bc_id_map: std.AutoHashMapUnmanaged(i64, PhysGroup) = .empty;
        try geo.readPhysicalNames(arena, &scan, &bc_id_map);

        // --- Map mesh entities to physical groups (format 4 only) ---
        // In format 2 every element carries its physical tag directly; in
        // format 4 elements only know their *entity*, and the entity carries
        // the physical tag.
        var ent_map: std.AutoHashMapUnmanaged(EntityKey, i64) = .empty;
        if (version == .v4) try readEntities(arena, &scan, &ent_map);

        // --- Read Mesh Vertex Locations ---
        var node_map: NodeMap = undefined;
        try geo.readNodes(arena, &scan, version, &node_map);

        // --- Read Element Connectivity ---
        try geo.readElements(arena, &scan, version, &bc_id_map, &ent_map, &node_map);
    }

    // ---- $MeshFormat ----

    const Version = enum { v2, v4 };

    fn readMeshFormat(scan: *Scanner) !Version {
        try scan.seekSection("$MeshFormat");
        var it = Scanner.fields(try scan.nextLine());
        const ver_str = it.next() orelse return error.MalformedMeshFile;

        // The ASCII/binary flag is the second field; we only handle ASCII.
        const file_type = try std.fmt.parseInt(u8, it.next() orelse "0", 10);
        if (file_type != 0) {
            report("Geo: binary Gmsh files are not supported\n", .{});
            return error.UnsupportedMeshFormat;
        }

        var ver_it = std.mem.splitScalar(u8, ver_str, '.');
        const major = std.fmt.parseInt(u32, ver_it.first(), 10) catch {
            report("Geo: unrecognized Gmsh format version '{s}'\n", .{ver_str});
            return error.UnsupportedMeshFormat;
        };
        const minor = std.fmt.parseInt(u32, ver_it.next() orelse "0", 10) catch 0;

        return switch (major) {
            2 => .v2,
            // 4.0 laid out `$Entities` differently than 4.1 did, so parsing it
            // as 4.1 would silently produce a garbled mesh.
            4 => if (minor >= 1) .v4 else {
                report("Geo: Gmsh format 4.0 is not supported; re-save as 4.1 or 2.2\n", .{});
                return error.UnsupportedMeshFormat;
            },
            else => {
                report("Geo: unsupported Gmsh format version '{s}'\n", .{ver_str});
                return error.UnsupportedMeshFormat;
            },
        };
    }

    // ---- $PhysicalNames ----

    /// Which Flurry region a Gmsh physical group maps to.
    const PhysGroup = union(enum) {
        /// The interior (solved) region
        fluid,
        /// Index into `bc_list` / `bc_names`
        bnd: usize,
    };

    /// Read `$PhysicalNames` and build the map from Gmsh physical tag to
    /// Flurry boundary index. Populates `n_gmsh_bnds`, `n_bounds`, `bc_list`,
    /// `bc_names` and `n_dims`.
    fn readPhysicalNames(
        geo: *Geo,
        arena: std.mem.Allocator,
        scan: *Scanner,
        bc_id_map: *std.AutoHashMapUnmanaged(i64, PhysGroup),
    ) !void {
        const gpa = geo.gpa;
        const mesh_bounds = &geo.config.boundary_conditions.mesh_bounds.fields;

        try scan.seekSection("$PhysicalNames");
        geo.n_gmsh_bnds = try std.fmt.parseInt(usize, Scanner.trim(try scan.nextLine()), 10);

        geo.n_bounds = 0;
        var found_fluid = false;

        for (0..geo.n_gmsh_bnds) |_| {
            const line = try scan.nextLine();
            var it = Scanner.fields(line);
            const bc_dim = try std.fmt.parseInt(usize, it.next() orelse return error.MalformedMeshFile, 10);
            const bc_tag = try std.fmt.parseInt(i64, it.next() orelse return error.MalformedMeshFile, 10);

            // The name is quoted and may contain spaces, so take everything
            // between the first and last quote rather than tokenizing.
            const q0 = std.mem.indexOfScalar(u8, line, '"') orelse return error.MalformedMeshFile;
            const q1 = std.mem.lastIndexOfScalar(u8, line, '"') orelse return error.MalformedMeshFile;
            if (q1 <= q0) return error.MalformedMeshFile;
            const bc_name = line[q0 + 1 .. q1];

            // Match against the input file case-insensitively
            const name_lower = try std.ascii.allocLowerString(arena, bc_name);

            if (std.mem.eql(u8, name_lower, "fluid")) {
                // The interior region: its dimension is the mesh dimension.
                geo.n_dims = bc_dim;
                found_fluid = true;
                if (bc_dim != geo.config.core.n_dims) {
                    report(
                        "Geo: WARNING: mesh 'FLUID' region is {d}D but config specifies n_dims = {d}\n",
                        .{ bc_dim, geo.config.core.n_dims },
                    );
                }
                try bc_id_map.put(arena, bc_tag, .fluid);
                continue;
            }

            // Map the Gmsh PhysicalName to the input-file-specified boundary condition
            const bc = mesh_bounds.get(name_lower) orelse {
                report(
                    "Geo: unrecognized mesh boundary: \"{s}\"\n" ++
                        "Boundary names in the input file must match those in the mesh file.\n",
                    .{bc_name},
                );
                return error.UnrecognizedMeshBoundary;
            };

            try bc_id_map.put(arena, bc_tag, .{ .bnd = geo.n_bounds });
            try geo.bc_list.append(gpa, bc);
            try geo.bc_names.append(gpa, try gpa.dupe(u8, name_lower));
            geo.n_bounds += 1;
        }

        if (!found_fluid) {
            report("Geo: no physical group named \"FLUID\" found in mesh file\n", .{});
            return error.NoFluidRegion;
        }
    }

    // ---- $Entities (format 4 only) ----

    const EntityKey = struct { dim: u8, tag: i64 };

    /// Build the (entity dim, entity tag) -> physical tag map. Entities with
    /// no physical tag are simply absent from the map; their elements are
    /// skipped when reading `$Elements`.
    fn readEntities(
        arena: std.mem.Allocator,
        scan: *Scanner,
        ent_map: *std.AutoHashMapUnmanaged(EntityKey, i64),
    ) !void {
        try scan.seekSection("$Entities");

        var hdr = Scanner.fields(try scan.nextLine());
        var counts: [4]usize = undefined;
        for (&counts) |*n| {
            n.* = try std.fmt.parseInt(usize, hdr.next() orelse return error.MalformedMeshFile, 10);
        }

        for (counts, 0..) |count, dim| {
            // Points list only their 3 coordinates; curves/surfaces/volumes
            // list a 6-value bounding box.
            const n_coords: usize = if (dim == 0) 3 else 6;
            for (0..count) |_| {
                var it = Scanner.fields(try scan.nextLine());
                const tag = try std.fmt.parseInt(i64, it.next() orelse return error.MalformedMeshFile, 10);
                for (0..n_coords) |_| _ = it.next() orelse return error.MalformedMeshFile;

                const n_phys = try std.fmt.parseInt(usize, it.next() orelse return error.MalformedMeshFile, 10);
                if (n_phys == 0) continue;

                // An entity may carry several physical tags; Flurry only
                // supports one region per entity, so take the first.
                const phys = try std.fmt.parseInt(i64, it.next() orelse return error.MalformedMeshFile, 10);
                try ent_map.put(arena, .{ .dim = @intCast(dim), .tag = tag }, phys);
            }
        }
    }

    // ---- $Nodes ----

    /// Translates Gmsh node tags into 0-based vertex indices. Gmsh tags are
    /// usually a contiguous 1-based run, in which case no table is needed.
    const NodeMap = union(enum) {
        contiguous: usize, // n_verts
        sparse: std.AutoHashMapUnmanaged(i64, usize),

        fn get(self: *const NodeMap, tag: i64) !usize {
            switch (self.*) {
                .contiguous => |n| {
                    if (tag < 1 or @as(usize, @intCast(tag)) > n) return error.UnknownNodeTag;
                    return @as(usize, @intCast(tag)) - 1;
                },
                .sparse => |*m| return m.get(tag) orelse error.UnknownNodeTag,
            }
        }
    };

    /// Read `$Nodes`, filling `n_verts` and `xv` and building the tag map.
    fn readNodes(
        geo: *Geo,
        arena: std.mem.Allocator,
        scan: *Scanner,
        version: Version,
        node_map: *NodeMap,
    ) !void {
        try scan.seekSection("$Nodes");

        var hdr = Scanner.fields(try scan.nextLine());
        const first = try std.fmt.parseInt(usize, hdr.next() orelse return error.MalformedMeshFile, 10);

        // Format 2: "numNodes".  Format 4: "numBlocks numNodes minTag maxTag".
        const n_blocks: usize = if (version == .v4) first else 1;
        geo.n_verts = if (version == .v4)
            try std.fmt.parseInt(usize, hdr.next() orelse return error.MalformedMeshFile, 10)
        else
            first;

        geo.xv = try Matrix(f64).init(geo.gpa, geo.n_verts, geo.n_dims, null);

        // Assume contiguous 1-based tags; fall back to a table if that breaks.
        node_map.* = .{ .contiguous = geo.n_verts };
        var sparse: std.AutoHashMapUnmanaged(i64, usize) = .empty;
        var is_sparse = false;

        var iv: usize = 0;
        for (0..n_blocks) |_| {
            const n_in_block: usize = if (version == .v4) blk: {
                // entityDim entityTag parametric numNodesInBlock
                var bh = Scanner.fields(try scan.nextLine());
                for (0..3) |_| _ = bh.next() orelse return error.MalformedMeshFile;
                break :blk try std.fmt.parseInt(usize, bh.next() orelse return error.MalformedMeshFile, 10);
            } else geo.n_verts;

            // Format 4 lists all tags in the block, then all coordinates.
            // Format 2 lists "tag x y z" per line.
            const block_start = iv;
            if (version == .v4) {
                for (0..n_in_block) |i| {
                    const tag = try std.fmt.parseInt(i64, Scanner.trim(try scan.nextLine()), 10);
                    if (tag != @as(i64, @intCast(block_start + i)) + 1) is_sparse = true;
                    try sparse.put(arena, tag, block_start + i);
                }
            }

            for (0..n_in_block) |i| {
                var it = Scanner.fields(try scan.nextLine());
                if (version == .v2) {
                    const tag = try std.fmt.parseInt(i64, it.next() orelse return error.MalformedMeshFile, 10);
                    if (tag != @as(i64, @intCast(block_start + i)) + 1) is_sparse = true;
                    try sparse.put(arena, tag, block_start + i);
                }
                // Gmsh always writes 3 coordinates; keep only the first n_dims.
                for (0..geo.n_dims) |d| {
                    const s = it.next() orelse return error.MalformedMeshFile;
                    geo.xv.at(block_start + i, d).* = try std.fmt.parseFloat(f64, s);
                }
            }
            iv += n_in_block;
        }

        if (iv != geo.n_verts) return error.MalformedMeshFile;
        if (is_sparse) node_map.* = .{ .sparse = sparse };
    }

    // ---- $Elements ----

    /// Read `$Elements`: interior elements become cells, elements belonging to
    /// a boundary physical group contribute their nodes to that boundary's
    /// point list.
    fn readElements(
        geo: *Geo,
        arena: std.mem.Allocator,
        scan: *Scanner,
        version: Version,
        bc_id_map: *const std.AutoHashMapUnmanaged(i64, PhysGroup),
        ent_map: *const std.AutoHashMapUnmanaged(EntityKey, i64),
        node_map: *const NodeMap,
    ) !void {
        const gpa = geo.gpa;

        try scan.seekSection("$Elements");

        // One sorted set of node indices per boundary
        const bound_points = try arena.alloc(NodeSet, geo.n_bounds);
        for (bound_points) |*s| s.* = .empty;

        // Cell vertex lists are gathered here first: the number of interior
        // cells (and the max vertices per cell) isn't known until we're done.
        var cell_verts: std.ArrayList([]const usize) = .empty;
        var max_nv: usize = 0;

        // Scratch buffers, sized for the largest element we support
        // (quintic hex, 216 nodes).
        var nodes_buf: [max_ele_nodes]usize = undefined;
        var vert_buf: [max_ele_nodes]usize = undefined;

        // Physical tags used by elements but never declared in
        // `$PhysicalNames`, tracked so each is only warned about once.
        var undeclared: std.AutoArrayHashMapUnmanaged(i64, void) = .empty;

        var hdr = Scanner.fields(try scan.nextLine());
        const first = try std.fmt.parseInt(usize, hdr.next() orelse return error.MalformedMeshFile, 10);

        // Format 2: "numElements".  Format 4: "numBlocks numElements minTag maxTag".
        const n_blocks: usize = if (version == .v4) first else 1;
        const n_eles_gmsh: usize = if (version == .v4)
            try std.fmt.parseInt(usize, hdr.next() orelse return error.MalformedMeshFile, 10)
        else
            first;

        for (0..n_blocks) |_| {
            // Format 4 declares the type and physical group once per block;
            // format 2 repeats them on every element line.
            var block_type: u32 = 0;
            var block_group: ?PhysGroup = null;
            const n_in_block: usize = if (version == .v4) blk: {
                // entityDim entityTag elementType numElementsInBlock
                var bh = Scanner.fields(try scan.nextLine());
                const dim = try std.fmt.parseInt(u8, bh.next() orelse return error.MalformedMeshFile, 10);
                const tag = try std.fmt.parseInt(i64, bh.next() orelse return error.MalformedMeshFile, 10);
                block_type = try std.fmt.parseInt(u32, bh.next() orelse return error.MalformedMeshFile, 10);
                const n = try std.fmt.parseInt(usize, bh.next() orelse return error.MalformedMeshFile, 10);

                // An entity with no physical tag at all belongs to no group,
                // which is normal and not worth warning about.
                if (ent_map.get(.{ .dim = dim, .tag = tag })) |phys| {
                    block_group = try resolveGroup(arena, bc_id_map, phys, &undeclared);
                }
                break :blk n;
            } else n_eles_gmsh;

            for (0..n_in_block) |_| {
                var it = Scanner.fields(try scan.nextLine());
                _ = it.next() orelse return error.MalformedMeshFile; // element tag

                var e_type = block_type;
                var group = block_group;
                if (version == .v2) {
                    e_type = try std.fmt.parseInt(u32, it.next() orelse return error.MalformedMeshFile, 10);
                    const n_tags = try std.fmt.parseInt(usize, it.next() orelse return error.MalformedMeshFile, 10);
                    // The first tag is the physical group; the rest are the
                    // elementary entity, partition info, etc.
                    for (0..n_tags) |t| {
                        const s = it.next() orelse return error.MalformedMeshFile;
                        if (t == 0) {
                            const phys = try std.fmt.parseInt(i64, s, 10);
                            group = try resolveGroup(arena, bc_id_map, phys, &undeclared);
                        }
                    }
                }

                // Elements outside any group we know about carry no
                // information for us.
                const g = group orelse continue;

                // The remaining fields on the line are node tags.
                var n_nodes: usize = 0;
                while (it.next()) |s| {
                    if (n_nodes == nodes_buf.len) return error.UnrecognizedElementType;
                    const tag = try std.fmt.parseInt(i64, s, 10);
                    nodes_buf[n_nodes] = try node_map.get(tag);
                    n_nodes += 1;
                }
                const nodes = nodes_buf[0..n_nodes];

                switch (g) {
                    .fluid => {
                        const spec = cellFromGmsh(e_type, nodes, &vert_buf) catch |err| {
                            report(
                                "Geo: Gmsh element type {d} not recognized as a cell\n",
                                .{e_type},
                            );
                            return err;
                        };
                        try geo.c2nv.append(gpa, spec.n_verts);
                        try geo.c2nf.append(gpa, spec.n_faces);
                        try geo.ctype.append(gpa, spec.ctype);
                        try cell_verts.append(arena, try arena.dupe(usize, vert_buf[0..spec.n_verts]));
                        max_nv = @max(max_nv, spec.n_verts);
                    },
                    .bnd => |ib| {
                        // Every node of a boundary element lies on that
                        // boundary -- including high-order edge/face nodes.
                        for (nodes) |v| try bound_points[ib].put(arena, v, {});
                    },
                }
            }
        }

        // --- Pack the gathered cell connectivity into c2v ---
        if (cell_verts.items.len == 0) {
            report("Geo: mesh contains no interior elements in the 'FLUID' region\n", .{});
            return error.NoInteriorCells;
        }
        geo.n_eles = cell_verts.items.len;
        geo.n_nodes_per_cell = max_nv;
        geo.c2v = try Matrix(usize).init(gpa, geo.n_eles, max_nv, null);
        for (cell_verts.items, 0..) |verts, ic| {
            for (verts, 0..) |v, j| geo.c2v.at(ic, j).* = v;
        }

        try geo.packBoundaryPoints(bound_points);

        report(
            "Geo: read {d} vertices, {d} cells, {d} boundaries\n",
            .{ geo.n_verts, geo.n_eles, geo.n_bounds },
        );
    }

    /// Pack one node set per boundary into `bnd_pts` / `n_bnd_pts`.
    ///
    /// `bnd_pts` is rectangular, so rows for boundaries with fewer points are
    /// zero-padded; `n_bnd_pts[i]` is the only reliable way to know where row
    /// `i` ends. The sets are consumed (their key arrays get sorted in place).
    fn packBoundaryPoints(geo: *Geo, bound_points: []NodeSet) !void {
        const gpa = geo.gpa;
        std.debug.assert(bound_points.len == geo.n_bounds);

        try geo.n_bnd_pts.resize(gpa, geo.n_bounds);
        var max_n_bnd_pts: usize = 0;
        for (bound_points, 0..) |*set, i| {
            geo.n_bnd_pts.items[i] = set.count();
            max_n_bnd_pts = @max(max_n_bnd_pts, set.count());
        }

        geo.bnd_pts = try Matrix(usize).init(gpa, geo.n_bounds, max_n_bnd_pts, null);
        for (bound_points, 0..) |*set, i| {
            const pts = set.keys();
            // Sorted, so the node ordering is independent of insertion order
            std.mem.sortUnstable(usize, pts, {}, std.sort.asc(usize));
            for (pts, 0..) |v, j| geo.bnd_pts.at(i, j).* = v;
        }
    }

    // ---- Cartesian mesh generation ----

    /// Generate a uniform Cartesian mesh from `config.create_mesh`, filling the
    /// same members `readGmsh` does: `xv`, `c2v`, `c2nv`, `c2nf`, `ctype`,
    /// `bc_list`, `bc_names`, `bnd_pts`, `n_bnd_pts`, plus `n_faces_per_bnd`
    /// and the `periodic_d*` box extents.
    pub fn createMesh(geo: *Geo) !void {
        const cm = geo.config.create_mesh orelse {
            report("Geo: createMesh called but no [create_mesh] config section is present\n", .{});
            return error.MissingCreateMeshConfig;
        };

        geo.n_dims = geo.config.core.n_dims;
        if (geo.n_dims != 2 and geo.n_dims != 3) return error.UnsupportedDimension;

        const grid: CartGrid = .init(cm, geo.n_dims);

        report(
            "Geo: Creating {d}x{d}x{d} cartesian mesh\n",
            .{ grid.nx, grid.ny, grid.nz },
        );

        // Box extents, needed later to match up periodic boundaries
        geo.periodic_dx = cm.xmax - cm.xmin;
        geo.periodic_dy = cm.ymax - cm.ymin;
        geo.periodic_dz = cm.zmax - cm.zmin;

        var arena_state: std.heap.ArenaAllocator = .init(geo.gpa);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        switch (geo.n_dims) {
            2 => try geo.createMesh2D(grid),
            3 => try geo.createMesh3D(grid),
            else => unreachable,
        }

        // --- Boundaries ---
        // The six (four in 2D) sides may share boundary conditions, so the
        // distinct BCs become the boundary list and each side is routed to its
        // entry in it.
        const bound_points = try geo.setupBoundaries(arena, cm);
        switch (geo.n_dims) {
            2 => try geo.boundaryFaces2D(arena, grid, cm, bound_points),
            3 => try geo.boundaryFaces3D(arena, grid, cm, bound_points),
            else => unreachable,
        }
        try geo.packBoundaryPoints(bound_points);

        report(
            "Geo: created {d} vertices, {d} cells, {d} boundaries\n",
            .{ geo.n_verts, geo.n_eles, geo.n_bounds },
        );
    }

    /// Vertices and cells of a 2D (quad) Cartesian mesh.
    fn createMesh2D(geo: *Geo, grid: CartGrid) !void {
        const gpa = geo.gpa;
        const nx = grid.nx;
        const ny = grid.ny;

        geo.n_verts = (nx + 1) * (ny + 1);
        geo.n_eles = nx * ny;
        geo.n_nodes_per_cell = 4;

        geo.xv = try Matrix(f64).init(gpa, geo.n_verts, 2, null);
        for (0..ny + 1) |j| {
            for (0..nx + 1) |i| {
                const iv = grid.vert(i, j, 0);
                geo.xv.at(iv, 0).* = grid.x(i);
                geo.xv.at(iv, 1).* = grid.y(j);
            }
        }

        try geo.c2nv.appendNTimes(gpa, 4, geo.n_eles);
        try geo.c2nf.appendNTimes(gpa, 4, geo.n_eles);
        try geo.ctype.appendNTimes(gpa, .quad, geo.n_eles);

        geo.c2v = try Matrix(usize).init(gpa, geo.n_eles, 4, null);
        var ic: usize = 0;
        // x-outer / y-inner, matching Flurry-cpp's cell numbering
        for (0..nx) |i| {
            for (0..ny) |j| {
                geo.c2v.at(ic, 0).* = grid.vert(i, j, 0);
                geo.c2v.at(ic, 1).* = grid.vert(i + 1, j, 0);
                geo.c2v.at(ic, 2).* = grid.vert(i + 1, j + 1, 0);
                geo.c2v.at(ic, 3).* = grid.vert(i, j + 1, 0);
                ic += 1;
            }
        }
        std.debug.assert(ic == geo.n_eles);
    }

    /// Vertices and cells of a 3D (hex) Cartesian mesh.
    fn createMesh3D(geo: *Geo, grid: CartGrid) !void {
        const gpa = geo.gpa;
        const nx = grid.nx;
        const ny = grid.ny;
        const nz = grid.nz;

        geo.n_verts = (nx + 1) * (ny + 1) * (nz + 1);
        geo.n_eles = nx * ny * nz;
        geo.n_nodes_per_cell = 8;

        geo.xv = try Matrix(f64).init(gpa, geo.n_verts, 3, null);
        for (0..nz + 1) |k| {
            for (0..ny + 1) |j| {
                for (0..nx + 1) |i| {
                    const iv = grid.vert(i, j, k);
                    geo.xv.at(iv, 0).* = grid.x(i);
                    geo.xv.at(iv, 1).* = grid.y(j);
                    geo.xv.at(iv, 2).* = grid.z(k);
                }
            }
        }

        try geo.c2nv.appendNTimes(gpa, 8, geo.n_eles);
        try geo.c2nf.appendNTimes(gpa, 6, geo.n_eles);
        try geo.ctype.appendNTimes(gpa, .hex, geo.n_eles);

        geo.c2v = try Matrix(usize).init(gpa, geo.n_eles, 8, null);
        var ic: usize = 0;
        // z-outer, then x, then y -- matching Flurry-cpp's cell numbering
        for (0..nz) |k| {
            for (0..nx) |i| {
                for (0..ny) |j| {
                    // Bottom face (z = k), then the same four at z = k+1
                    geo.c2v.at(ic, 0).* = grid.vert(i, j, k);
                    geo.c2v.at(ic, 1).* = grid.vert(i + 1, j, k);
                    geo.c2v.at(ic, 2).* = grid.vert(i + 1, j + 1, k);
                    geo.c2v.at(ic, 3).* = grid.vert(i, j + 1, k);

                    geo.c2v.at(ic, 4).* = grid.vert(i, j, k + 1);
                    geo.c2v.at(ic, 5).* = grid.vert(i + 1, j, k + 1);
                    geo.c2v.at(ic, 6).* = grid.vert(i + 1, j + 1, k + 1);
                    geo.c2v.at(ic, 7).* = grid.vert(i, j + 1, k + 1);
                    ic += 1;
                }
            }
        }
        std.debug.assert(ic == geo.n_eles);
    }

    /// Collapse the per-side boundary conditions into the distinct-BC list
    /// (`bc_list`, `bc_names`, `n_bounds`) and return one empty node set per
    /// boundary, arena-owned, for the face loops to fill.
    fn setupBoundaries(
        geo: *Geo,
        arena: std.mem.Allocator,
        cm: cfg.CreateMeshConfig,
    ) ![]NodeSet {
        const gpa = geo.gpa;

        // Order follows Flurry-cpp; it only affects the pre-sort arrangement.
        // front/back exist in 3D only, so they come last and get sliced off.
        var sides: [6]cfg.BoundaryCondition = .{
            cm.bc_bottom, cm.bc_right, cm.bc_top,
            cm.bc_left,   cm.bc_front, cm.bc_back,
        };

        // Sort and drop duplicates: sides sharing a BC share a boundary.
        const list = sides[0..if (geo.n_dims == 3) @as(usize, 6) else 4];
        std.mem.sortUnstable(cfg.BoundaryCondition, list, {}, struct {
            fn lessThan(_: void, a: cfg.BoundaryCondition, b: cfg.BoundaryCondition) bool {
                return @backingInt(a) < @backingInt(b);
            }
        }.lessThan);

        var n_bounds: usize = 0;
        for (list) |bc| {
            if (n_bounds > 0 and list[n_bounds - 1] == bc) continue;
            list[n_bounds] = bc;
            n_bounds += 1;
        }
        geo.n_bounds = n_bounds;

        try geo.bc_list.appendSlice(gpa, list[0..n_bounds]);
        for (list[0..n_bounds]) |bc| {
            // No mesh file to take names from, so the BC tag is the name.
            try geo.bc_names.append(gpa, try gpa.dupe(u8, @tagName(bc)));
        }

        try geo.n_faces_per_bnd.appendNTimes(gpa, 0, n_bounds);

        const bound_points = try arena.alloc(NodeSet, n_bounds);
        for (bound_points) |*s| s.* = .empty;
        return bound_points;
    }

    /// Index into `bc_list` for a side's boundary condition.
    fn boundIndex(geo: *const Geo, bc: cfg.BoundaryCondition) usize {
        return std.mem.indexOfScalar(cfg.BoundaryCondition, geo.bc_list.items, bc).?;
    }

    /// Record the nodes of one boundary face against its boundary.
    fn addBoundaryFace(
        geo: *Geo,
        arena: std.mem.Allocator,
        bound_points: []NodeSet,
        bc: cfg.BoundaryCondition,
        nodes: []const usize,
    ) !void {
        const ib = geo.boundIndex(bc);
        for (nodes) |v| try bound_points[ib].put(arena, v, {});
        geo.n_faces_per_bnd.items[ib] += 1;
    }

    /// Boundary edges of a 2D mesh. `bottom`/`top` are y = ymin/ymax and
    /// `left`/`right` are x = xmin/xmax; each edge is wound so the interior
    /// lies to its left.
    fn boundaryFaces2D(
        geo: *Geo,
        arena: std.mem.Allocator,
        grid: CartGrid,
        cm: cfg.CreateMeshConfig,
        bound_points: []NodeSet,
    ) !void {
        const nx = grid.nx;
        const ny = grid.ny;

        for (0..nx) |ix| {
            try geo.addBoundaryFace(arena, bound_points, cm.bc_bottom, &.{
                grid.vert(ix, 0, 0), grid.vert(ix + 1, 0, 0),
            });
            try geo.addBoundaryFace(arena, bound_points, cm.bc_top, &.{
                grid.vert(ix + 1, ny, 0), grid.vert(ix, ny, 0),
            });
        }

        for (0..ny) |iy| {
            try geo.addBoundaryFace(arena, bound_points, cm.bc_left, &.{
                grid.vert(0, iy + 1, 0), grid.vert(0, iy, 0),
            });
            try geo.addBoundaryFace(arena, bound_points, cm.bc_right, &.{
                grid.vert(nx, iy, 0), grid.vert(nx, iy + 1, 0),
            });
        }
    }

    /// Boundary faces of a 3D mesh. Note the axes differ from the 2D case:
    /// `bottom`/`top` are z = zmin/zmax, `left`/`right` are x = xmin/xmax and
    /// `back`/`front` are y = ymin/ymax.
    fn boundaryFaces3D(
        geo: *Geo,
        arena: std.mem.Allocator,
        grid: CartGrid,
        cm: cfg.CreateMeshConfig,
        bound_points: []NodeSet,
    ) !void {
        const nx = grid.nx;
        const ny = grid.ny;
        const nz = grid.nz;

        // z = zmin / z = zmax
        for (0..nx) |ix| {
            for (0..ny) |iy| {
                try geo.addBoundaryFace(arena, bound_points, cm.bc_bottom, &.{
                    grid.vert(ix, iy, 0),         grid.vert(ix + 1, iy, 0),
                    grid.vert(ix + 1, iy + 1, 0), grid.vert(ix, iy + 1, 0),
                });
                try geo.addBoundaryFace(arena, bound_points, cm.bc_top, &.{
                    grid.vert(ix, iy, nz),         grid.vert(ix + 1, iy, nz),
                    grid.vert(ix + 1, iy + 1, nz), grid.vert(ix, iy + 1, nz),
                });
            }
        }

        // x = xmin / x = xmax
        for (0..nz) |iz| {
            for (0..ny) |iy| {
                try geo.addBoundaryFace(arena, bound_points, cm.bc_left, &.{
                    grid.vert(0, iy, iz),         grid.vert(0, iy, iz + 1),
                    grid.vert(0, iy + 1, iz + 1), grid.vert(0, iy + 1, iz),
                });
                try geo.addBoundaryFace(arena, bound_points, cm.bc_right, &.{
                    grid.vert(nx, iy, iz),         grid.vert(nx, iy, iz + 1),
                    grid.vert(nx, iy + 1, iz + 1), grid.vert(nx, iy + 1, iz),
                });
            }
        }

        // y = ymin / y = ymax
        for (0..nz) |iz| {
            for (0..nx) |ix| {
                try geo.addBoundaryFace(arena, bound_points, cm.bc_back, &.{
                    grid.vert(ix, 0, iz),         grid.vert(ix, 0, iz + 1),
                    grid.vert(ix + 1, 0, iz + 1), grid.vert(ix + 1, 0, iz),
                });
                try geo.addBoundaryFace(arena, bound_points, cm.bc_front, &.{
                    grid.vert(ix, ny, iz),         grid.vert(ix, ny, iz + 1),
                    grid.vert(ix + 1, ny, iz + 1), grid.vert(ix + 1, ny, iz),
                });
            }
        }
    }

    // ---- Connectivity processing ----

    /// Derive face and neighbour connectivity from `c2v`: `f2v`, `f2nv`,
    /// `face_type`, `int_faces`, `bnd_faces`, `c2f`, `c2b`, `c2c`, `f2c`, the
    /// boundary-face matching (`bc_type`, `bc_id`, `bc_faces`), and in 3D the
    /// edge list (`e2v`, `n_edges`).
    ///
    /// Must be called after `readGmsh` or `createMesh`.
    ///
    /// TODO: Flurry-cpp's `processConnectivity` continues on to
    /// `processConnExtra` (v2v/v2e and the bounding box),
    /// `processPeriodicBoundaries` and `matchMPIFaces`; none of those are
    /// ported yet.
    pub fn processConnectivity(geo: *Geo) !void {
        report("Geo: Processing element connectivity\n", .{});

        // Validate before doing any work, so nothing below has to cope with a
        // dimension or cell type it was not written for.
        switch (geo.n_dims) {
            2 => for (geo.ctype.items) |ct| {
                if (ct == .hex) return error.UnsupportedCellType;
            },
            3 => for (geo.ctype.items) |ct| {
                if (ct != .hex) return error.UnsupportedCellType;
            },
            else => return error.UnsupportedDimension,
        }

        var arena_state: std.heap.ArenaAllocator = .init(geo.gpa);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        // Periodic faces are interior faces whose two sides sit on opposite
        // ends of the domain. Giving each periodic vertex pair a shared
        // canonical ID makes them merge during face construction.
        const pm = try geo.buildPeriodicVertexMap(arena);
        const canon: ?[]const usize = if (pm) |m| m.canon else null;

        switch (geo.n_dims) {
            2 => try geo.processConn2D(arena, canon),
            3 => try geo.processConn3D(arena, canon),
            else => return error.UnsupportedDimension,
        }

        if (pm) |m| {
            try geo.checkPeriodicFaceConsistency(arena, m);

            // A periodic face that found no partner stayed a boundary face, and
            // there is no boundary condition that can stand in for one.
            for (geo.bc_type.items) |bc| {
                if (bc == .periodic) {
                    report("Geo: a periodic boundary face found no partner\n", .{});
                    return error.UnmatchedPeriodicFace;
                }
            }
        }
    }

    /// Build the global flux point map, given how many flux points an element
    /// puts on each face (`order + 1` for a tensor-product element).
    ///
    /// Call after `processConnectivity`. The element's local flux points must be
    /// stored face by face, in the same face order as `localFaces`, each face
    /// traversed counter-clockwise -- which is what `eles/quads.zig` does.
    ///
    /// Two cells sharing a face both traverse it counter-clockwise *from their
    /// own side*, so they walk it in opposite directions: the left cell's flux
    /// point `j` sits on top of the right cell's `n_fpts_per_face - 1 - j`. That
    /// holds exactly because the 1D flux point distribution is symmetric about
    /// the face midpoint. `Solver` checks it geometrically once it knows the
    /// physical flux point locations.
    ///
    /// ZEFR relies on the same reversal, then runs `orient_fpts` to re-sort by
    /// physical coordinate for the cases where it does not hold (3D face
    /// rotations). A 2D conforming mesh needs no such fixup.
    pub fn setupGlobalFpts(geo: *Geo, n_fpts_per_face: usize) !void {
        const gpa = geo.gpa;

        if (geo.n_faces == 0) return error.ConnectivityNotProcessed;

        var max_nf: usize = 0;
        for (geo.c2nf.items) |nf| max_nf = @max(max_nf, nf);

        geo.n_fpts_per_face = n_fpts_per_face;
        geo.n_fpts_per_ele = max_nf * n_fpts_per_face;
        geo.n_gfpts_int = geo.n_int_faces * n_fpts_per_face;
        geo.n_gfpts_bnd = geo.n_bnd_faces * n_fpts_per_face;
        geo.n_gfpts = geo.n_gfpts_int + geo.n_gfpts_bnd;

        geo.fpt2gfpt = try Matrix(usize).init(gpa, geo.n_fpts_per_ele, geo.n_eles, null);
        @memset(geo.fpt2gfpt.data, none);
        geo.fpt2gfpt_slot = try Matrix(u8).init(gpa, geo.n_fpts_per_ele, geo.n_eles, null);
        try geo.gfpt2bnd.resize(gpa, geo.n_gfpts_bnd);
        try geo.gfpt2face.resize(gpa, geo.n_gfpts);

        var gfpt: usize = 0;

        // --- Interior faces: both sides, second one reversed ---
        for (geo.int_faces.items) |ff| {
            const ic_l = geo.f2c.get(ff, 0);
            const ic_r = geo.f2c.get(ff, 1);
            const lf_l = geo.localFaceOf(ic_l, ff).?;
            const lf_r = geo.localFaceOf(ic_r, ff).?;

            const fpt0_l = lf_l * n_fpts_per_face;
            const fpt0_r = lf_r * n_fpts_per_face;

            for (0..n_fpts_per_face) |j| {
                const jr = n_fpts_per_face - 1 - j;
                geo.fpt2gfpt.at(fpt0_l + j, ic_l).* = gfpt;
                geo.fpt2gfpt_slot.at(fpt0_l + j, ic_l).* = 0;
                geo.fpt2gfpt.at(fpt0_r + jr, ic_r).* = gfpt;
                geo.fpt2gfpt_slot.at(fpt0_r + jr, ic_r).* = 1;

                geo.gfpt2face.items[gfpt] = ff;
                gfpt += 1;
            }
        }

        // --- Boundary faces: left side only; the right state comes from the BC ---
        for (geo.bnd_faces.items, 0..) |ff, i| {
            const ic = geo.f2c.get(ff, 0);
            const lf = geo.localFaceOf(ic, ff).?;
            const fpt0 = lf * n_fpts_per_face;

            for (0..n_fpts_per_face) |j| {
                geo.fpt2gfpt.at(fpt0 + j, ic).* = gfpt;
                geo.fpt2gfpt_slot.at(fpt0 + j, ic).* = 0;

                geo.gfpt2bnd.items[gfpt - geo.n_gfpts_int] = geo.bc_id.items[i];
                geo.gfpt2face.items[gfpt] = ff;
                gfpt += 1;
            }
        }

        std.debug.assert(gfpt == geo.n_gfpts);

        report(
            "Geo: {d} global flux points ({d} interior, {d} boundary)\n",
            .{ geo.n_gfpts, geo.n_gfpts_int, geo.n_gfpts_bnd },
        );
    }

    /// Which of cell `ic`'s faces is global face `ff`.
    pub fn localFaceOf(geo: *const Geo, ic: usize, ff: usize) ?usize {
        for (0..geo.c2nf.items[ic]) |j| {
            if (geo.c2f.get(ic, j) == ff) return j;
        }
        return null;
    }

    /// Which axis directions the mesh is genuinely periodic in.
    ///
    /// Vertex positions alone cannot answer this. In a mesh periodic in x but
    /// walled in y, the corners `(xmin, ymin)` and `(xmin, ymax)` both sit on
    /// the periodic boundary and are separated by exactly the bounding-box y
    /// extent, so a raw pairwise sweep identifies them across a direction that
    /// is not periodic at all.
    ///
    /// Boundary *faces* settle it: direction `d` is periodic exactly when the
    /// periodic boundary includes faces lying in the plane normal to `d` at
    /// both ends of the domain. Boundary faces are the cell faces claimed by
    /// exactly one cell, which is the same counting `buildFaces` does -- but it
    /// has to happen before that, since the answer decides how faces are keyed.
    fn periodicDirections(
        geo: *Geo,
        arena: std.mem.Allocator,
        on_periodic: *const std.AutoArrayHashMapUnmanaged(usize, void),
        lo: [3]f64,
        hi: [3]f64,
        tol: f64,
    ) ![3]bool {
        const nd = geo.n_dims;

        var counts: std.AutoArrayHashMapUnmanaged(FaceKey, usize) = .empty;
        const min_verts: usize = if (nd == 3) 3 else 2;
        var verts: [max_face_verts]usize = undefined;

        for (0..geo.n_eles) |ic| {
            for (localFaces(geo.ctype.items[ic])) |lf| {
                for (lf, 0..) |lv, i| verts[i] = geo.c2v.get(ic, lv);
                const key = FaceKey.init(verts[0..lf.len]);
                if (key.n_verts < min_verts) continue;

                const gop = try counts.getOrPut(arena, key);
                if (!gop.found_existing) gop.value_ptr.* = 0;
                gop.value_ptr.* += 1;
            }
        }

        var at_lo: [3]bool = .{ false, false, false };
        var at_hi: [3]bool = .{ false, false, false };

        for (counts.keys(), counts.values()) |key, n_cells| {
            if (n_cells != 1) continue; // interior face

            const fv = key.verts[0..key.n_verts];
            for (fv) |iv| {
                if (!on_periodic.contains(iv)) break;
            } else {
                // A planar axis-aligned face is normal to whichever direction
                // it has no extent in, and a periodic plane spans the domain,
                // so it sits at one end of the bounding box.
                for (0..nd) |d| {
                    var f_lo = std.math.inf(f64);
                    var f_hi = -std.math.inf(f64);
                    for (fv) |iv| {
                        f_lo = @min(f_lo, geo.xv.get(iv, d));
                        f_hi = @max(f_hi, geo.xv.get(iv, d));
                    }
                    if (f_hi - f_lo >= tol) continue;
                    if (@abs(f_lo - lo[d]) < tol) at_lo[d] = true;
                    if (@abs(f_lo - hi[d]) < tol) at_hi[d] = true;
                }
            }
        }

        var dirs: [3]bool = .{ false, false, false };
        for (0..nd) |d| dirs[d] = at_lo[d] and at_hi[d];
        return dirs;
    }

    /// Give every vertex on a periodic boundary a canonical ID shared with its
    /// image(s) across the domain, so that `buildFaces` sees a periodic face and
    /// its partner as the same face.
    ///
    /// Returns `null` when the mesh has no periodic boundary, in which case face
    /// construction uses real vertex IDs directly.
    ///
    /// A vertex `b` is the periodic image of `a` when, for some non-empty set of
    /// directions, their separation equals that direction's period and is zero
    /// in every other direction. That covers domain corners, which are images of
    /// one another in two or three directions at once -- hence the union-find,
    /// so all four corners of a doubly-periodic box collapse to one ID.
    ///
    /// Flurry-cpp instead paired faces after the fact, enumerating the four 2D
    /// orientation cases explicitly and comparing face normals and centroids in
    /// 3D, then rewired `f2c`/`c2f`/`c2b` and left the absorbed faces as holes
    /// in the face list. Matching at the vertex level is dimension-generic and
    /// leaves the face arrays with no gaps.
    fn buildPeriodicVertexMap(geo: *Geo, arena: std.mem.Allocator) !?PeriodicMap {
        const nd = geo.n_dims;

        // Which vertices sit on a periodic boundary?
        var on_periodic: std.AutoArrayHashMapUnmanaged(usize, void) = .empty;
        for (geo.bc_list.items, 0..) |bc, bnd| {
            if (bc != .periodic) continue;
            for (0..geo.n_bnd_pts.items[bnd]) |j| {
                try on_periodic.put(arena, geo.bnd_pts.get(bnd, j), {});
            }
        }
        if (on_periodic.count() == 0) return null;

        // Periods: whatever the mesh recorded, else the bounding box extents,
        // which is the standard assumption for a periodic mesh read from a file.
        var lo: [3]f64 = .{ std.math.inf(f64), std.math.inf(f64), std.math.inf(f64) };
        var hi: [3]f64 = .{ -std.math.inf(f64), -std.math.inf(f64), -std.math.inf(f64) };
        for (0..geo.n_verts) |iv| {
            for (0..nd) |d| {
                lo[d] = @min(lo[d], geo.xv.get(iv, d));
                hi[d] = @max(hi[d], geo.xv.get(iv, d));
            }
        }

        var period: [3]f64 = .{ geo.periodic_dx, geo.periodic_dy, geo.periodic_dz };
        var diag: f64 = 0.0;
        for (0..nd) |d| {
            if (period[d] == 0.0) period[d] = hi[d] - lo[d];
            diag += (hi[d] - lo[d]) * (hi[d] - lo[d]);
        }
        diag = @sqrt(diag);

        // Relative to the domain size, so the same tolerance works whatever
        // units the mesh is in.
        const tol = 1e-8 * diag;

        // A zero period means "no shift is allowed here", which is what the
        // matching below needs for a direction that is not actually periodic:
        // only vertices at the same coordinate in `d` may be identified.
        const dirs = try geo.periodicDirections(arena, &on_periodic, lo, hi, tol);
        var any_dir = false;
        for (0..nd) |d| {
            if (dirs[d]) any_dir = true else period[d] = 0.0;
        }
        if (!any_dir) {
            report("Geo: a periodic boundary is declared but spans no periodic direction\n", .{});
            return error.UnmatchedPeriodicFace;
        }

        const parent = try arena.alloc(usize, geo.n_verts);
        for (parent, 0..) |*p, i| p.* = i;

        // Pairwise over periodic boundary vertices only, which is a thin shell
        // of the mesh. A spatial hash would make this linear if it ever matters.
        const pts = on_periodic.keys();
        for (pts, 0..) |a, i| {
            for (pts[i + 1 ..]) |b| {
                var shifted = false;
                var matches = true;
                for (0..nd) |d| {
                    const sep = @abs(geo.xv.get(b, d) - geo.xv.get(a, d));
                    if (sep < tol) continue; // same position in this direction
                    if (@abs(sep - period[d]) < tol) {
                        shifted = true;
                        continue;
                    }
                    matches = false;
                    break;
                }
                if (matches and shifted) unite(parent, a, b);
            }
        }

        const canon = try arena.alloc(usize, geo.n_verts);
        var n_merged: usize = 0;
        for (canon, 0..) |*c, iv| {
            c.* = find(parent, iv);
            if (c.* != iv) n_merged += 1;
        }

        report("Geo: merged {d} periodic vertex images\n", .{n_merged});
        return .{ .canon = canon, .period = period, .tol = tol };
    }

    /// Verify that the two cells sharing a face agree on where it is, allowing
    /// for one periodic offset.
    ///
    /// Merging periodic vertices identifies faces by vertex set, which is exact
    /// as long as no two distinct faces end up with the same set. That fails when
    /// a periodic direction spans only two cells: the boundary line normal to it
    /// closes into a two-edge loop whose edges span the same vertex pair. Without
    /// this check the result is either a non-manifold face or, worse, a pair of
    /// unrelated faces silently fused into one.
    fn checkPeriodicFaceConsistency(geo: *Geo, arena: std.mem.Allocator, pm: PeriodicMap) !void {
        const nd = geo.n_dims;

        const seen = try arena.alloc(?[3]f64, geo.n_faces);
        @memset(seen, null);

        for (0..geo.n_eles) |ic| {
            for (localFaces(geo.ctype.items[ic]), 0..) |lf, j| {
                const ff = geo.c2f.get(ic, j);
                if (ff == none) continue;

                // Centroid of this cell's copy of the face
                var c: [3]f64 = .{ 0, 0, 0 };
                for (lf) |lv| {
                    const iv = geo.c2v.get(ic, lv);
                    for (0..nd) |d| c[d] += geo.xv.get(iv, d);
                }
                for (0..nd) |d| c[d] /= @floatFromInt(lf.len);

                const other = seen[ff] orelse {
                    seen[ff] = c;
                    continue;
                };

                // The two copies must coincide, or be one period apart
                for (0..nd) |d| {
                    const sep = @abs(c[d] - other[d]);
                    if (sep < pm.tol) continue;
                    if (@abs(sep - pm.period[d]) < pm.tol) continue;

                    report(
                        "Geo: periodic merging fused two different faces into one.\n" ++
                            "A periodic direction needs at least three cells across it.\n",
                        .{},
                    );
                    return error.PeriodicDirectionTooThin;
                }
            }
        }
    }

    /// 2D: the faces are the cell edges.
    fn processConn2D(geo: *Geo, arena: std.mem.Allocator, canon: ?[]const usize) !void {
        var faces: FaceBuilder = .empty;
        try geo.buildFaces(arena, &faces, canon);
        try geo.finishConn(arena, &faces, canon != null);
    }

    /// 3D: the faces come from each cell type's local face table, and the edge
    /// list is accumulated alongside them.
    fn processConn3D(geo: *Geo, arena: std.mem.Allocator, canon: ?[]const usize) !void {
        var faces: FaceBuilder = .empty;
        try geo.buildFaces(arena, &faces, canon);
        try geo.buildEdges(arena);
        try geo.finishConn(arena, &faces, canon != null);
    }

    /// Walk every cell face, registering it in `faces` and recording which
    /// global face each cell slot maps to in `c2f`.
    fn buildFaces(
        geo: *Geo,
        arena: std.mem.Allocator,
        faces: *FaceBuilder,
        canon: ?[]const usize,
    ) !void {
        const gpa = geo.gpa;

        var max_nf: usize = 0;
        for (geo.c2nf.items) |nf| max_nf = @max(max_nf, nf);

        geo.c2f = try Matrix(usize).init(gpa, geo.n_eles, max_nf, null);
        @memset(geo.c2f.data, none);

        var verts: [max_face_verts]usize = undefined;
        var canon_verts: [max_face_verts]usize = undefined;

        // A 2D face (edge) needs two distinct vertices, a 3D face three; with
        // fewer it has collapsed to a line or a point and is not a face.
        const min_verts: usize = if (geo.n_dims == 3) 3 else 2;

        for (0..geo.n_eles) |ic| {
            const local = localFaces(geo.ctype.items[ic]);
            // A collapsed cell (a tet stored as a hex, say) reports fewer faces
            // than its type has. Flurry-cpp silently processed only the first
            // c2nf of them, quietly dropping the rest.
            if (local.len != geo.c2nf.items[ic]) {
                report(
                    "Geo: cell {d} is a {s} with {d} faces, but its type has {d}\n",
                    .{ ic, @tagName(geo.ctype.items[ic]), geo.c2nf.items[ic], local.len },
                );
                return error.InconsistentCellType;
            }

            for (local, 0..) |lf, j| {
                for (lf, 0..) |lv, i| verts[i] = geo.c2v.get(ic, lv);
                const real = FaceKey.init(verts[0..lf.len]);
                if (real.n_verts < min_verts) continue;

                // Under periodicity a face is identified by its canonical
                // vertices, so it and its image across the domain register as
                // one face with two cells -- and everything downstream
                // (int/bnd classification, c2c, c2b, the global flux points)
                // then treats it as interior with no further special casing.
                const key = if (canon) |c| blk: {
                    for (lf, 0..) |lv, i| canon_verts[i] = c[geo.c2v.get(ic, lv)];
                    break :blk FaceKey.init(canon_verts[0..lf.len]);
                } else real;

                geo.c2f.at(ic, j).* = try faces.add(arena, key, real);
            }
        }
    }

    /// Collect the unique mesh edges (3D only; in 2D the faces *are* the edges).
    ///
    /// Flurry-cpp derived these by pairing up each face's *sorted* vertex list,
    /// which does not describe the face's edges at all -- sorting destroys the
    /// cyclic order, so two of the four pairs came out as diagonals. Edges are
    /// taken from the local face table's cyclic order here instead.
    fn buildEdges(geo: *Geo, arena: std.mem.Allocator) !void {
        var edges: std.AutoArrayHashMapUnmanaged([2]usize, void) = .empty;

        for (0..geo.n_eles) |ic| {
            for (localFaces(geo.ctype.items[ic])) |lf| {
                for (lf, 0..) |lv, i| {
                    const iv1 = geo.c2v.get(ic, lv);
                    const iv2 = geo.c2v.get(ic, lf[(i + 1) % lf.len]);
                    if (iv1 == iv2) continue; // collapsed edge
                    const key: [2]usize = if (iv1 < iv2) .{ iv1, iv2 } else .{ iv2, iv1 };
                    try edges.put(arena, key, {});
                }
            }
        }

        geo.n_edges = edges.count();
        geo.e2v = try Matrix(usize).init(geo.gpa, geo.n_edges, 2, null);
        for (edges.keys(), 0..) |key, ie| {
            geo.e2v.at(ie, 0).* = key[0];
            geo.e2v.at(ie, 1).* = key[1];
        }
    }

    /// Everything after face enumeration, shared by 2D and 3D: pack `f2v`,
    /// classify each face, match boundary faces to boundaries, and link cells
    /// to faces and to each other.
    fn finishConn(
        geo: *Geo,
        arena: std.mem.Allocator,
        faces: *FaceBuilder,
        periodic: bool,
    ) !void {
        const gpa = geo.gpa;

        // --- f2v / f2nv ---
        // Rows hold the face's vertices in ascending order, not in an oriented
        // traversal; matching duplicates is all they are used for.
        geo.n_faces = faces.keys.count();
        var max_fnv: usize = 0;
        for (faces.verts.items) |key| max_fnv = @max(max_fnv, key.n_verts);

        geo.f2v = try Matrix(usize).init(gpa, geo.n_faces, max_fnv, null);
        @memset(geo.f2v.data, none);
        try geo.f2nv.resize(gpa, geo.n_faces);
        for (faces.verts.items, 0..) |key, ff| {
            geo.f2nv.items[ff] = key.n_verts;
            for (0..key.n_verts) |j| geo.f2v.at(ff, j).* = key.verts[j];
        }

        // --- Internal vs. boundary faces ---
        // MPI faces still count as boundary faces at this stage.
        try geo.face_type.appendNTimes(gpa, .internal, geo.n_faces);
        geo.n_int_faces = 0;
        geo.n_bnd_faces = 0;
        geo.n_mpi_faces = 0;

        for (faces.n_cells.items, 0..) |n_cells, ff| {
            switch (n_cells) {
                1 => {
                    geo.face_type.items[ff] = .boundary;
                    try geo.bnd_faces.append(gpa, ff);
                    geo.n_bnd_faces += 1;
                },
                2 => {
                    try geo.int_faces.append(gpa, ff);
                    geo.n_int_faces += 1;
                },
                else => {
                    if (periodic) {
                        // Far and away the likeliest cause on a periodic mesh
                        report(
                            "Geo: periodic merging fused {d} cells onto face {d}.\n" ++
                                "A periodic direction needs at least three cells across it.\n",
                            .{ n_cells, ff },
                        );
                        return error.PeriodicDirectionTooThin;
                    }
                    report("Geo: face {d} is shared by {d} cells\n", .{ ff, n_cells });
                    return error.NonManifoldFace;
                },
            }
        }

        try geo.matchBoundaryFaces(arena);
        try geo.linkCellsAndFaces();
    }

    /// Assign each boundary face to a boundary: every one of its vertices must
    /// lie on that boundary. Fills `bc_type`, `bc_id` and `bc_faces`.
    fn matchBoundaryFaces(geo: *Geo, arena: std.mem.Allocator) !void {
        const gpa = geo.gpa;

        try geo.bc_type.appendNTimes(gpa, .none, geo.n_bnd_faces);
        try geo.bc_id.appendNTimes(gpa, none, geo.n_bnd_faces);

        // Face IDs per boundary, gathered before packing into bc_faces
        const per_bnd = try arena.alloc(std.ArrayList(usize), geo.n_bounds);
        for (per_bnd) |*l| l.* = .empty;

        var n_unmatched: usize = 0;
        for (geo.bnd_faces.items, 0..) |ff, i| {
            const matched = for (0..geo.n_bounds) |bnd| {
                if (geo.faceOnBoundary(ff, bnd)) break bnd;
            } else null;

            if (matched) |bnd| {
                geo.bc_type.items[i] = geo.bc_list.items[bnd];
                geo.bc_id.items[i] = bnd;
                try per_bnd[bnd].append(arena, ff);
            } else {
                n_unmatched += 1;
            }
        }

        if (n_unmatched > 0) {
            // In a parallel run these would be the processor boundaries; with
            // no MPI support yet, they mean the mesh has boundary faces that
            // no declared boundary covers.
            report(
                "Geo: WARNING: {d} of {d} boundary faces matched no boundary\n",
                .{ n_unmatched, geo.n_bnd_faces },
            );
        }

        // --- Pack bc_faces, and record how many faces each boundary has ---
        try geo.n_faces_per_bnd.resize(gpa, geo.n_bounds);
        try geo.bc_faces.resize(gpa, geo.n_bounds);
        for (per_bnd, 0..) |*list, bnd| {
            geo.n_faces_per_bnd.items[bnd] = list.items.len;
            var mat = try Matrix(usize).init(gpa, list.items.len, geo.f2v.cols, null);
            @memset(mat.data, none);
            for (list.items, 0..) |ff, row| {
                for (0..geo.f2nv.items[ff]) |j| mat.at(row, j).* = geo.f2v.get(ff, j);
            }
            geo.bc_faces.items[bnd] = mat;
        }
    }

    /// Whether every vertex of face `ff` lies on boundary `bnd`.
    fn faceOnBoundary(geo: *const Geo, ff: usize, bnd: usize) bool {
        // bnd_pts rows are sorted and n_bnd_pts gives the exact extent, so the
        // zero padding beyond it is never searched.
        const row = geo.bnd_pts.data[bnd * geo.bnd_pts.stride ..];
        const pts = row[0..geo.n_bnd_pts.items[bnd]];

        for (0..geo.f2nv.items[ff]) |j| {
            const iv = geo.f2v.get(ff, j);
            if (std.sort.binarySearch(usize, pts, iv, orderUsize) == null) return false;
        }
        return true;
    }

    /// Fill `f2c` (the up-to-two cells on each face), `c2c` (face-neighbour
    /// cells) and `c2b` (whether each cell face is on a boundary).
    fn linkCellsAndFaces(geo: *Geo) !void {
        const gpa = geo.gpa;
        const max_nf = geo.c2f.cols;

        geo.c2b = try Matrix(usize).init(gpa, geo.n_eles, max_nf, null);
        geo.c2c = try Matrix(usize).init(gpa, geo.n_eles, max_nf, null);
        @memset(geo.c2c.data, none);
        geo.f2c = try Matrix(usize).init(gpa, geo.n_faces, 2, null);
        @memset(geo.f2c.data, none);

        for (0..geo.n_eles) |ic| {
            for (0..geo.c2nf.items[ic]) |j| {
                const ff = geo.c2f.get(ic, j);
                if (ff == none) continue; // collapsed face

                geo.c2b.at(ic, j).* = @intFromBool(@backingInt(geo.face_type.items[ff]) > 0);

                const left = geo.f2c.get(ff, 0);
                if (left == none) {
                    // First cell seen on this face goes on the left
                    geo.f2c.at(ff, 0).* = ic;
                    continue;
                }
                // A collapsed cell can reach the same face twice; it is not its
                // own neighbour.
                if (left == ic) continue;

                geo.f2c.at(ff, 1).* = ic;

                // Both cells are now neighbours across face ff. The left cell
                // has a lower ID, so its c2f row is already filled in.
                const fid2 = for (0..geo.c2nf.items[left]) |k| {
                    if (geo.c2f.get(left, k) == ff) break k;
                } else unreachable;

                geo.c2c.at(ic, j).* = left;
                geo.c2c.at(left, fid2).* = ic;
            }
        }
    }
};

fn orderUsize(key: usize, mid: usize) std.math.Order {
    return std.math.order(key, mid);
}

/// Faces are identified by their corner vertices only, so four is the most any
/// supported cell type needs (a hex face); a 2D "face" is an edge, with two.
const max_face_verts = 4;

/// A face's identity: its corner vertices, sorted ascending and de-duplicated,
/// with the unused slots set to `none`.
///
/// Sorting makes the same face reachable from either adjoining cell regardless
/// of the order that cell lists it in, and de-duplicating turns a collapsed
/// quad face into the triangle it geometrically is. Flurry-cpp instead replaced
/// repeated vertices with -1 before sorting, which left the sentinel *inside*
/// the key and made a 3-vertex face compare equal to the first three entries of
/// an unrelated 4-vertex one.
const FaceKey = struct {
    verts: [max_face_verts]usize,
    n_verts: usize,

    fn init(verts: []const usize) FaceKey {
        std.debug.assert(verts.len >= 2 and verts.len <= max_face_verts);

        var key: FaceKey = .{ .verts = @splat(none), .n_verts = 0 };
        @memcpy(key.verts[0..verts.len], verts);
        std.mem.sortUnstable(usize, key.verts[0..verts.len], {}, std.sort.asc(usize));

        // Collapse runs of equal vertices, then blank the freed slots so the
        // key stays comparable by value.
        for (key.verts[0..verts.len]) |v| {
            if (key.n_verts > 0 and key.verts[key.n_verts - 1] == v) continue;
            key.verts[key.n_verts] = v;
            key.n_verts += 1;
        }
        @memset(key.verts[key.n_verts..], none);
        return key;
    }
};

/// Builds the unique global face list, counting how many cells touch each.
///
/// Flurry-cpp matched cell faces against the global list with a linear scan per
/// face, making its 3D connectivity O(n_eles * n_faces); a hash lookup keeps it
/// linear.
const FaceBuilder = struct {
    /// Identity used for de-duplication. Under periodicity this holds canonical
    /// vertex IDs, so a face and its periodic image collapse into one.
    keys: std.AutoArrayHashMapUnmanaged(FaceKey, void),

    /// The face's *actual* vertices, as first registered. Kept separately
    /// because `f2v` has to describe real geometry -- boundary matching compares
    /// it against `bnd_pts`, which is in real vertex IDs.
    verts: std.ArrayList(FaceKey),

    n_cells: std.ArrayList(usize),

    const empty: FaceBuilder = .{ .keys = .empty, .verts = .empty, .n_cells = .empty };

    /// Global ID of `key`, registering it the first time it is seen. Face IDs
    /// are assignment order, which the hash map preserves.
    fn add(fb: *FaceBuilder, arena: std.mem.Allocator, key: FaceKey, real: FaceKey) !usize {
        const gop = try fb.keys.getOrPut(arena, key);
        if (!gop.found_existing) {
            try fb.n_cells.append(arena, 0);
            try fb.verts.append(arena, real);
        }
        fb.n_cells.items[gop.index] += 1;
        return gop.index;
    }
};

/// The vertex identification that makes periodic faces merge, plus the periods
/// it was derived from.
const PeriodicMap = struct {
    /// `canon[iv]` is the representative of `iv`'s periodic equivalence class
    canon: []const usize,
    /// Domain period in each direction
    period: [3]f64,
    /// Absolute position tolerance, relative to the domain size
    tol: f64,
};

/// Union-find root of `i`, with path compression.
fn find(parent: []usize, i: usize) usize {
    var root = i;
    while (parent[root] != root) root = parent[root];
    var walk = i;
    while (parent[walk] != root) {
        const next = parent[walk];
        parent[walk] = root;
        walk = next;
    }
    return root;
}

/// Merge two vertices' equivalence classes, keeping the lower ID as the root so
/// the canonical representative is deterministic.
fn unite(parent: []usize, a: usize, b: usize) void {
    const ra = find(parent, a);
    const rb = find(parent, b);
    if (ra == rb) return;
    if (ra < rb) parent[rb] = ra else parent[ra] = rb;
}

/// Corner-vertex indices of each face of a cell, in cyclic order. In 2D a
/// "face" is an edge.
///
/// The hex ordering is Flurry-cpp's `ct2fv[HEX]`, which carries a
/// "FIX ORDERING FOR FUTURE USE" note upstream -- it is not necessarily the
/// face ordering the solver's flux points assume.
fn localFaces(ct: CellType) []const []const u8 {
    return switch (ct) {
        .tri => &.{ &.{ 0, 1 }, &.{ 1, 2 }, &.{ 2, 0 } },
        .quad => &.{ &.{ 0, 1 }, &.{ 1, 2 }, &.{ 2, 3 }, &.{ 3, 0 } },
        .hex => &.{
            &.{ 0, 1, 2, 3 }, // Bottom (zmin)
            &.{ 4, 5, 6, 7 }, // Top    (zmax)
            &.{ 3, 0, 4, 7 }, // Left   (xmin)
            &.{ 2, 1, 5, 6 }, // Right  (xmax)
            &.{ 1, 0, 4, 5 }, // Front  (ymin)
            &.{ 3, 2, 6, 7 }, // Back   (ymax)
        },
    };
}

/// Uniform Cartesian grid geometry: cell counts, origin and spacing.
///
/// Vertices are numbered x-fastest then y then z, so `vert(i, j, k)` is the
/// single source of truth for the layout that both the cell connectivity and
/// the boundary face loops index against.
const CartGrid = struct {
    nx: usize,
    ny: usize,
    nz: usize,
    xmin: f64,
    ymin: f64,
    zmin: f64,
    dx: f64,
    dy: f64,
    dz: f64,

    fn init(cm: cfg.CreateMeshConfig, n_dims: usize) CartGrid {
        // A 2D mesh is one cell thick, which also keeps n_eles = nx*ny*nz.
        const nz: usize = if (n_dims == 2) 1 else cm.nz;
        return .{
            .nx = cm.nx,
            .ny = cm.ny,
            .nz = nz,
            .xmin = cm.xmin,
            .ymin = cm.ymin,
            .zmin = cm.zmin,
            .dx = (cm.xmax - cm.xmin) / @as(f64, @floatFromInt(cm.nx)),
            .dy = (cm.ymax - cm.ymin) / @as(f64, @floatFromInt(cm.ny)),
            .dz = (cm.zmax - cm.zmin) / @as(f64, @floatFromInt(nz)),
        };
    }

    /// 0-based vertex index at grid position (i, j, k)
    fn vert(g: CartGrid, i: usize, j: usize, k: usize) usize {
        std.debug.assert(i <= g.nx and j <= g.ny and k <= g.nz);
        return i + (g.nx + 1) * (j + (g.ny + 1) * k);
    }

    fn x(g: CartGrid, i: usize) f64 {
        return g.xmin + @as(f64, @floatFromInt(i)) * g.dx;
    }

    fn y(g: CartGrid, j: usize) f64 {
        return g.ymin + @as(f64, @floatFromInt(j)) * g.dy;
    }

    fn z(g: CartGrid, k: usize) f64 {
        return g.zmin + @as(f64, @floatFromInt(k)) * g.dz;
    }
};

/// A sorted-on-demand set of node indices
const NodeSet = std.AutoArrayHashMapUnmanaged(usize, void);

/// Largest element we support: quintic (216-node) hexahedron
const max_ele_nodes = 216;

const CellSpec = struct {
    n_verts: usize,
    n_faces: usize,
    ctype: CellType,
};

/// Translate a Gmsh element's node list into Flurry's cell vertex ordering.
///
/// `nodes` holds the element's nodes as 0-based vertex indices in Gmsh order;
/// the result is written to `out` and the vertex/face counts returned.
///
/// For Gmsh node ordering, see:
/// http://geuz.org/gmsh/doc/texinfo/gmsh.html#Node-ordering
fn cellFromGmsh(e_type: u32, nodes: []const usize, out: []usize) !CellSpec {
    // Element types whose Gmsh ordering already matches ours: a straight copy
    // of `n` nodes.
    const direct: ?CellSpec = switch (e_type) {
        2 => .{ .n_verts = 3, .n_faces = 3, .ctype = .tri }, // linear triangle
        3 => .{ .n_verts = 4, .n_faces = 4, .ctype = .quad }, // linear quad
        16 => .{ .n_verts = 8, .n_faces = 4, .ctype = .quad }, // quadratic serendipity quad
        10 => .{ .n_verts = 9, .n_faces = 4, .ctype = .quad }, // quadratic Lagrange quad
        36 => .{ .n_verts = 16, .n_faces = 4, .ctype = .quad }, // cubic quad
        37 => .{ .n_verts = 25, .n_faces = 4, .ctype = .quad }, // quartic quad
        38 => .{ .n_verts = 36, .n_faces = 4, .ctype = .quad }, // quintic quad
        47 => .{ .n_verts = 49, .n_faces = 4, .ctype = .quad }, // 6th-order quad
        48 => .{ .n_verts = 64, .n_faces = 4, .ctype = .quad }, // 7th-order quad
        49 => .{ .n_verts = 81, .n_faces = 4, .ctype = .quad }, // 8th-order quad
        50 => .{ .n_verts = 100, .n_faces = 4, .ctype = .quad }, // 9th-order quad
        51 => .{ .n_verts = 121, .n_faces = 4, .ctype = .quad }, // 10th-order quad
        5 => .{ .n_verts = 8, .n_faces = 6, .ctype = .hex }, // linear hex
        12 => .{ .n_verts = 27, .n_faces = 6, .ctype = .hex }, // quadratic Lagrange hex
        92 => .{ .n_verts = 64, .n_faces = 6, .ctype = .hex }, // cubic hex
        93 => .{ .n_verts = 125, .n_faces = 6, .ctype = .hex }, // quartic hex
        94 => .{ .n_verts = 216, .n_faces = 6, .ctype = .hex }, // quintic hex
        else => null,
    };
    if (direct) |spec| {
        if (nodes.len < spec.n_verts) return error.MalformedMeshFile;
        @memcpy(out[0..spec.n_verts], nodes[0..spec.n_verts]);
        return spec;
    }

    switch (e_type) {
        9 => {
            // Quadratic triangle -> quadratic (8-node) quad, by collapsing
            // corner 3 onto corner 2. Gmsh order is v0 v1 v2 e01 e12 e20.
            if (nodes.len < 6) return error.MalformedMeshFile;
            out[0] = nodes[0];
            out[1] = nodes[1];
            out[2] = nodes[2];
            out[3] = nodes[2]; // collapsed corner
            out[4] = nodes[3]; // e01
            out[5] = nodes[4]; // e12
            out[6] = nodes[2]; // collapsed edge
            out[7] = nodes[5]; // e20
            return .{ .n_verts = 8, .n_faces = 4, .ctype = .quad };
        },

        17 => {
            // Quadratic (20-node serendipity) hex. The corner nodes agree,
            // but Gmsh orders the 12 edge nodes differently than we do.
            if (nodes.len < 20) return error.MalformedMeshFile;
            @memcpy(out[0..8], nodes[0..8]);
            const edge_map = [12]usize{ 8, 11, 12, 9, 13, 10, 14, 15, 16, 19, 17, 18 };
            for (edge_map, 8..) |dst, src| out[dst] = nodes[src];
            return .{ .n_verts = 20, .n_faces = 6, .ctype = .hex };
        },

        4 => {
            // Linear tetrahedron, read as a collapsed-face hex.
            // TODO: tets are not yet properly supported downstream.
            if (nodes.len < 4) return error.MalformedMeshFile;
            out[0] = nodes[0];
            out[1] = nodes[1];
            out[2] = nodes[2];
            out[3] = nodes[2];
            out[4] = nodes[3];
            out[5] = nodes[3];
            out[6] = nodes[3];
            out[7] = nodes[3];
            return .{ .n_verts = 8, .n_faces = 4, .ctype = .hex };
        },

        6 => {
            // Linear prism, read as a collapsed-face hex
            if (nodes.len < 6) return error.MalformedMeshFile;
            out[0] = nodes[0];
            out[1] = nodes[1];
            out[2] = nodes[2];
            out[3] = nodes[2];
            out[4] = nodes[3];
            out[5] = nodes[4];
            out[6] = nodes[5];
            out[7] = nodes[5];
            return .{ .n_verts = 8, .n_faces = 6, .ctype = .hex };
        },

        else => return error.UnrecognizedElementType,
    }
}

/// Look up the Flurry region for a Gmsh physical tag.
///
/// A tag that elements reference but `$PhysicalNames` never declares means the
/// mesh file is internally inconsistent -- those elements get dropped, so warn
/// once per tag rather than silently producing a mesh with pieces missing.
fn resolveGroup(
    arena: std.mem.Allocator,
    bc_id_map: *const std.AutoHashMapUnmanaged(i64, Geo.PhysGroup),
    phys: i64,
    undeclared: *std.AutoArrayHashMapUnmanaged(i64, void),
) !?Geo.PhysGroup {
    if (bc_id_map.get(phys)) |group| return group;

    const gop = try undeclared.getOrPut(arena, phys);
    if (!gop.found_existing) {
        report(
            "Geo: WARNING: skipping elements with physical tag {d}, " ++
                "which is not declared in $PhysicalNames\n",
            .{phys},
        );
    }
    return null;
}

/// Diagnostic output for the mesh reader. Silenced under `zig build test` so
/// the expected-error cases don't spam the test log.
fn report(comptime fmt: []const u8, args: anytype) void {
    if (@import("builtin").is_test) return;
    std.debug.print(fmt, args);
}

/// Minimal line/section scanner over an in-memory Gmsh file.
///
/// Gmsh's ASCII format writes one entity per line, so line-oriented reading
/// handles every section uniformly -- including the variable-length node lists
/// on element lines, which a fixed per-type node count would get wrong for
/// high-order elements.
const Scanner = struct {
    src: []const u8,
    pos: usize = 0,

    /// Move the cursor to just past the line holding `tag`, searching from the
    /// start of the file (sections may appear in any order).
    fn seekSection(scan: *Scanner, tag: []const u8) !void {
        scan.pos = 0;
        while (scan.nextLine()) |line| {
            if (std.mem.eql(u8, trim(line), tag)) return;
        } else |_| {}
        report("Geo: '{s}' tag not found in Gmsh file!\n", .{tag});
        return error.MissingSection;
    }

    /// Next non-blank line, with the trailing newline (and any '\r') removed.
    fn nextLine(scan: *Scanner) ![]const u8 {
        while (scan.pos < scan.src.len) {
            const end = std.mem.indexOfScalarPos(u8, scan.src, scan.pos, '\n') orelse scan.src.len;
            const line = scan.src[scan.pos..end];
            scan.pos = @min(end + 1, scan.src.len);
            if (trim(line).len > 0) return line;
        }
        return error.MalformedMeshFile;
    }

    fn trim(line: []const u8) []const u8 {
        return std.mem.trim(u8, line, " \t\r");
    }

    fn fields(line: []const u8) std.mem.TokenIterator(u8, .any) {
        return std.mem.tokenizeAny(u8, line, " \t\r");
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// A single quad with two boundary edges, in Gmsh ASCII format 2.2.
const mesh_v2 =
    \\$MeshFormat
    \\2.2 0 8
    \\$EndMeshFormat
    \\$PhysicalNames
    \\3
    \\1 1 "Wall"
    \\1 2 "Inlet"
    \\2 3 "FLUID"
    \\$EndPhysicalNames
    \\$Nodes
    \\4
    \\1 0 0 0
    \\2 1 0 0
    \\3 1 1 0
    \\4 0 1 0
    \\$EndNodes
    \\$Elements
    \\3
    \\1 1 2 1 1 1 2
    \\2 1 2 2 2 3 4
    \\3 3 2 3 1 1 2 3 4
    \\$EndElements
;

/// The same mesh in format 4.1, with deliberately sparse node tags so the
/// tag -> index remapping is exercised.
const mesh_v4 =
    \\$MeshFormat
    \\4.1 0 8
    \\$EndMeshFormat
    \\$PhysicalNames
    \\3
    \\1 7 "Wall"
    \\1 8 "Inlet"
    \\2 9 "FLUID"
    \\$EndPhysicalNames
    \\$Entities
    \\0 2 1 0
    \\1 0 0 0 1 0 0 1 7 0
    \\2 0 1 0 1 1 0 1 8 0
    \\1 0 0 0 1 1 0 1 9 0
    \\$EndEntities
    \\$Nodes
    \\1 4 100 400
    \\2 1 0 4
    \\100
    \\200
    \\300
    \\400
    \\0 0 0
    \\1 0 0
    \\1 1 0
    \\0 1 0
    \\$EndNodes
    \\$Elements
    \\3 3 1 3
    \\1 1 1 1
    \\1 100 200
    \\1 2 1 1
    \\2 300 400
    \\2 1 3 1
    \\3 100 200 300 400
    \\$EndElements
;

/// Build a `Geo` whose `mesh_bounds` cover the test meshes below. `gpa` owns
/// everything `parseGmsh` produces (so leak checking exercises `deinit`);
/// `arena` owns the config's scratch data.
fn testGeo(gpa: std.mem.Allocator, arena: std.mem.Allocator, config: *cfg.Config) !Geo {
    config.core = .{ .n_dims = 2, .mesh_file = "", .order = 1 };
    config.boundary_conditions = .{};
    try config.boundary_conditions.mesh_bounds.fields.put(arena, "wall", .slip_wall);
    try config.boundary_conditions.mesh_bounds.fields.put(arena, "inlet", .sup_in);
    return .{ .gpa = gpa, .io = undefined, .config = config.* };
}

fn expectSingleQuadMesh(geo: *const Geo) !void {
    const t = std.testing;

    try t.expectEqual(@as(usize, 2), geo.n_dims);
    try t.expectEqual(@as(usize, 4), geo.n_verts);
    try t.expectEqual(@as(usize, 1), geo.n_eles);
    try t.expectEqual(@as(usize, 2), geo.n_bounds);
    try t.expectEqual(@as(usize, 4), geo.n_nodes_per_cell);

    // Vertices, with the unused z coordinate dropped
    const xv_expect = [4][2]f64{ .{ 0, 0 }, .{ 1, 0 }, .{ 1, 1 }, .{ 0, 1 } };
    for (xv_expect, 0..) |xy, i| {
        try t.expectEqual(xy[0], geo.xv.get(i, 0));
        try t.expectEqual(xy[1], geo.xv.get(i, 1));
    }

    // The single cell: a linear quad, 0-indexed
    try t.expectEqual(CellType.quad, geo.ctype.items[0]);
    try t.expectEqual(@as(usize, 4), geo.c2nv.items[0]);
    try t.expectEqual(@as(usize, 4), geo.c2nf.items[0]);
    for (0..4) |j| try t.expectEqual(j, geo.c2v.get(0, j));

    // Boundaries, in $PhysicalNames order
    try t.expectEqualStrings("wall", geo.bc_names.items[0]);
    try t.expectEqualStrings("inlet", geo.bc_names.items[1]);
    try t.expectEqual(cfg.BoundaryCondition.slip_wall, geo.bc_list.items[0]);
    try t.expectEqual(cfg.BoundaryCondition.sup_in, geo.bc_list.items[1]);

    try t.expectEqualSlices(usize, &.{ 2, 2 }, geo.n_bnd_pts.items);
    try t.expectEqual(@as(usize, 0), geo.bnd_pts.get(0, 0));
    try t.expectEqual(@as(usize, 1), geo.bnd_pts.get(0, 1));
    try t.expectEqual(@as(usize, 2), geo.bnd_pts.get(1, 0));
    try t.expectEqual(@as(usize, 3), geo.bnd_pts.get(1, 1));
}

test "readGmsh: format 2.2" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();

    var config: cfg.Config = undefined;
    var geo = try testGeo(gpa, arena.allocator(), &config);
    defer geo.deinit();

    try geo.parseGmsh(arena.allocator(), mesh_v2);
    try expectSingleQuadMesh(&geo);
}

test "readGmsh: format 4.1 with sparse node tags" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();

    var config: cfg.Config = undefined;
    var geo = try testGeo(gpa, arena.allocator(), &config);
    defer geo.deinit();

    try geo.parseGmsh(arena.allocator(), mesh_v4);
    try expectSingleQuadMesh(&geo);
}

test "readGmsh: unrecognized boundary name is an error" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();

    var config: cfg.Config = undefined;
    var geo = try testGeo(gpa, arena.allocator(), &config);
    defer geo.deinit();

    // "inlet" is in mesh_bounds; drop it so the mesh no longer matches.
    _ = config.boundary_conditions.mesh_bounds.fields.swapRemove("inlet");
    geo.config = config;

    try std.testing.expectError(error.UnrecognizedMeshBoundary, geo.parseGmsh(arena.allocator(), mesh_v2));
}

test "readGmsh: missing section is an error" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();

    var config: cfg.Config = undefined;
    var geo = try testGeo(gpa, arena.allocator(), &config);
    defer geo.deinit();

    const truncated = mesh_v2[0..std.mem.indexOf(u8, mesh_v2, "$Elements").?];
    try std.testing.expectError(error.MissingSection, geo.parseGmsh(arena.allocator(), truncated));
}

test cellFromGmsh {
    const t = std.testing;
    var out: [max_ele_nodes]usize = undefined;

    // Linear quad: straight copy
    {
        const spec = try cellFromGmsh(3, &.{ 10, 11, 12, 13 }, &out);
        try t.expectEqual(CellSpec{ .n_verts = 4, .n_faces = 4, .ctype = .quad }, spec);
        try t.expectEqualSlices(usize, &.{ 10, 11, 12, 13 }, out[0..4]);
    }

    // Quadratic triangle -> 8-node quad with corner 3 collapsed onto corner 2
    {
        const spec = try cellFromGmsh(9, &.{ 0, 1, 2, 3, 4, 5 }, &out);
        try t.expectEqual(CellSpec{ .n_verts = 8, .n_faces = 4, .ctype = .quad }, spec);
        try t.expectEqualSlices(usize, &.{ 0, 1, 2, 2, 3, 4, 2, 5 }, out[0..8]);
    }

    // Linear prism -> collapsed-face hex
    {
        const spec = try cellFromGmsh(6, &.{ 0, 1, 2, 3, 4, 5 }, &out);
        try t.expectEqual(CellSpec{ .n_verts = 8, .n_faces = 6, .ctype = .hex }, spec);
        try t.expectEqualSlices(usize, &.{ 0, 1, 2, 2, 3, 4, 5, 5 }, out[0..8]);
    }

    // A truncated node list must be rejected rather than read past the end
    try t.expectError(error.MalformedMeshFile, cellFromGmsh(3, &.{ 0, 1 }, &out));
    try t.expectError(error.UnrecognizedElementType, cellFromGmsh(999, &.{ 0, 1 }, &out));
}

// ---- createMesh ----

/// A `Geo` set up for Cartesian mesh generation. `gpa` owns everything
/// `createMesh` produces, so leak checking exercises `deinit`.
fn testCreateGeo(gpa: std.mem.Allocator, config: *cfg.Config, n_dims: u8, cm: cfg.CreateMeshConfig) Geo {
    config.core = .{ .n_dims = n_dims, .mesh_file = "", .order = 1 };
    config.boundary_conditions = .{};
    config.create_mesh = cm;
    return .{ .gpa = gpa, .io = undefined, .config = config.* };
}

/// Row `ib` of `bnd_pts`, trimmed to the entries that are actually populated.
fn bndPts(geo: *const Geo, ib: usize) []const usize {
    const row = geo.bnd_pts.data[ib * geo.bnd_pts.stride ..];
    return row[0..geo.n_bnd_pts.items[ib]];
}

test "createMesh: 2D, one BC per side" {
    const t = std.testing;
    const gpa = t.allocator;

    var config: cfg.Config = undefined;
    var geo = testCreateGeo(gpa, &config, 2, .{
        .nx = 2,
        .ny = 2,
        .xmin = 0.0,
        .xmax = 2.0,
        .ymin = 0.0,
        .ymax = 4.0,
        .bc_bottom = .slip_wall,
        .bc_right = .sup_out,
        .bc_top = .symmetry,
        .bc_left = .sup_in,
    });
    defer geo.deinit();

    try geo.createMesh();

    try t.expectEqual(@as(usize, 2), geo.n_dims);
    try t.expectEqual(@as(usize, 9), geo.n_verts);
    try t.expectEqual(@as(usize, 4), geo.n_eles);
    try t.expectEqual(@as(usize, 4), geo.n_nodes_per_cell);

    // Box extents, for periodic face matching
    try t.expectEqual(@as(f64, 2.0), geo.periodic_dx);
    try t.expectEqual(@as(f64, 4.0), geo.periodic_dy);

    // Vertices: x fastest, so dx = 1 and dy = 2
    const xv_expect = [9][2]f64{
        .{ 0, 0 }, .{ 1, 0 }, .{ 2, 0 },
        .{ 0, 2 }, .{ 1, 2 }, .{ 2, 2 },
        .{ 0, 4 }, .{ 1, 4 }, .{ 2, 4 },
    };
    for (xv_expect, 0..) |xy, iv| {
        try t.expectEqual(xy[0], geo.xv.get(iv, 0));
        try t.expectEqual(xy[1], geo.xv.get(iv, 1));
    }

    // Cells are numbered x-outer / y-inner, counter-clockwise within each
    const c2v_expect = [4][4]usize{
        .{ 0, 1, 4, 3 }, .{ 3, 4, 7, 6 },
        .{ 1, 2, 5, 4 }, .{ 4, 5, 8, 7 },
    };
    for (c2v_expect, 0..) |verts, ic| {
        try t.expectEqual(CellType.quad, geo.ctype.items[ic]);
        try t.expectEqual(@as(usize, 4), geo.c2nv.items[ic]);
        try t.expectEqual(@as(usize, 4), geo.c2nf.items[ic]);
        for (verts, 0..) |v, j| try t.expectEqual(v, geo.c2v.get(ic, j));
    }

    // Distinct BCs, sorted by enum value: sup_in < sup_out < slip_wall < symmetry
    try t.expectEqual(@as(usize, 4), geo.n_bounds);
    try t.expectEqualSlices(cfg.BoundaryCondition, &.{
        .sup_in, .sup_out, .slip_wall, .symmetry,
    }, geo.bc_list.items);
    try t.expectEqualStrings("sup_in", geo.bc_names.items[0]);
    try t.expectEqualStrings("symmetry", geo.bc_names.items[3]);

    try t.expectEqualSlices(usize, &.{ 0, 3, 6 }, bndPts(&geo, 0)); // left,   x = 0
    try t.expectEqualSlices(usize, &.{ 2, 5, 8 }, bndPts(&geo, 1)); // right,  x = 2
    try t.expectEqualSlices(usize, &.{ 0, 1, 2 }, bndPts(&geo, 2)); // bottom, y = 0
    // Vertex 0 is absent here. Flurry-cpp sorted the zero-padded `bnd_pts`
    // row and de-duplicated it, which folded the padding into a spurious
    // vertex 0 on every boundary that did not already contain it.
    try t.expectEqualSlices(usize, &.{ 6, 7, 8 }, bndPts(&geo, 3)); // top,    y = 4

    try t.expectEqualSlices(usize, &.{ 2, 2, 2, 2 }, geo.n_faces_per_bnd.items);
}

test "createMesh: 2D, all sides share one BC" {
    const t = std.testing;
    const gpa = t.allocator;

    var config: cfg.Config = undefined;
    var geo = testCreateGeo(gpa, &config, 2, .{
        .nx = 2,
        .ny = 2,
        .xmin = 0.0,
        .xmax = 2.0,
        .ymin = 0.0,
        .ymax = 2.0,
        // all four sides default to .periodic
    });
    defer geo.deinit();

    try geo.createMesh();

    // Sides sharing a BC collapse into a single boundary
    try t.expectEqual(@as(usize, 1), geo.n_bounds);
    try t.expectEqualSlices(cfg.BoundaryCondition, &.{.periodic}, geo.bc_list.items);

    // Every vertex except the center one (4) lies on the perimeter
    try t.expectEqualSlices(usize, &.{ 0, 1, 2, 3, 5, 6, 7, 8 }, bndPts(&geo, 0));
    try t.expectEqualSlices(usize, &.{8}, geo.n_faces_per_bnd.items);
}

test "createMesh: 3D, one BC per side" {
    const t = std.testing;
    const gpa = t.allocator;

    var config: cfg.Config = undefined;
    var geo = testCreateGeo(gpa, &config, 3, .{
        .nx = 1,
        .ny = 1,
        .nz = 1,
        .xmin = 0.0,
        .xmax = 1.0,
        .ymin = 0.0,
        .ymax = 1.0,
        .zmin = 0.0,
        .zmax = 1.0,
        .bc_bottom = .slip_wall,
        .bc_right = .sup_out,
        .bc_top = .symmetry,
        .bc_left = .sup_in,
        .bc_front = .characteristic,
        .bc_back = .adiabatic_noslip,
    });
    defer geo.deinit();

    try geo.createMesh();

    try t.expectEqual(@as(usize, 3), geo.n_dims);
    try t.expectEqual(@as(usize, 8), geo.n_verts);
    try t.expectEqual(@as(usize, 1), geo.n_eles);
    try t.expectEqual(@as(usize, 8), geo.n_nodes_per_cell);
    try t.expectEqual(@as(f64, 1.0), geo.periodic_dz);

    // The single hex, in Gmsh linear-hex vertex order
    try t.expectEqual(CellType.hex, geo.ctype.items[0]);
    try t.expectEqual(@as(usize, 6), geo.c2nf.items[0]);
    for ([8]usize{ 0, 1, 3, 2, 4, 5, 7, 6 }, 0..) |v, j| {
        try t.expectEqual(v, geo.c2v.get(0, j));
    }

    // Unit cube corners, x fastest then y then z
    try t.expectEqual(@as(f64, 0.0), geo.xv.get(0, 2));
    try t.expectEqual(@as(f64, 1.0), geo.xv.get(7, 0));
    try t.expectEqual(@as(f64, 1.0), geo.xv.get(7, 1));
    try t.expectEqual(@as(f64, 1.0), geo.xv.get(7, 2));

    // char < sup_in < sup_out < slip_wall < adiabatic_noslip < symmetry
    try t.expectEqual(@as(usize, 6), geo.n_bounds);
    try t.expectEqualSlices(cfg.BoundaryCondition, &.{
        .characteristic, .sup_in, .sup_out, .slip_wall, .adiabatic_noslip, .symmetry,
    }, geo.bc_list.items);

    // In 3D bottom/top are z, left/right are x, back/front are y
    try t.expectEqualSlices(usize, &.{ 2, 3, 6, 7 }, bndPts(&geo, 0)); // front,  y = 1
    try t.expectEqualSlices(usize, &.{ 0, 2, 4, 6 }, bndPts(&geo, 1)); // left,   x = 0
    try t.expectEqualSlices(usize, &.{ 1, 3, 5, 7 }, bndPts(&geo, 2)); // right,  x = 1
    try t.expectEqualSlices(usize, &.{ 0, 1, 2, 3 }, bndPts(&geo, 3)); // bottom, z = 0
    try t.expectEqualSlices(usize, &.{ 0, 1, 4, 5 }, bndPts(&geo, 4)); // back,   y = 0
    try t.expectEqualSlices(usize, &.{ 4, 5, 6, 7 }, bndPts(&geo, 5)); // top,    z = 1

    try t.expectEqualSlices(usize, &.{ 1, 1, 1, 1, 1, 1 }, geo.n_faces_per_bnd.items);
}

test "createMesh: cell and vertex counts" {
    const t = std.testing;
    const gpa = t.allocator;

    // 2D ignores nz entirely: the mesh is one cell thick
    {
        var config: cfg.Config = undefined;
        var geo = testCreateGeo(gpa, &config, 2, .{ .nx = 3, .ny = 4, .nz = 99 });
        defer geo.deinit();
        try geo.createMesh();
        try t.expectEqual(@as(usize, 4 * 5), geo.n_verts);
        try t.expectEqual(@as(usize, 3 * 4), geo.n_eles);
    }

    {
        var config: cfg.Config = undefined;
        var geo = testCreateGeo(gpa, &config, 3, .{ .nx = 3, .ny = 4, .nz = 5 });
        defer geo.deinit();
        try geo.createMesh();
        try t.expectEqual(@as(usize, 4 * 5 * 6), geo.n_verts);
        try t.expectEqual(@as(usize, 3 * 4 * 5), geo.n_eles);

        // Every vertex must be referenced by some cell, and no index may run
        // past the end of xv -- catches any slip in the vertex numbering.
        var seen = try gpa.alloc(bool, geo.n_verts);
        defer gpa.free(seen);
        @memset(seen, false);
        for (0..geo.n_eles) |ic| {
            for (0..8) |j| {
                const v = geo.c2v.get(ic, j);
                try t.expect(v < geo.n_verts);
                seen[v] = true;
            }
        }
        for (seen) |s| try t.expect(s);
    }
}

test "createMesh: missing config section is an error" {
    const gpa = std.testing.allocator;

    var config: cfg.Config = undefined;
    config.core = .{ .n_dims = 2, .mesh_file = "", .order = 1 };
    config.create_mesh = null;

    var geo: Geo = .{ .gpa = gpa, .io = undefined, .config = config };
    defer geo.deinit();

    try std.testing.expectError(error.MissingCreateMeshConfig, geo.createMesh());
}

// ---- processConnectivity ----

/// Row `ic` of `c2f` / `c2b` / `c2c`, trimmed to the cell's own face count.
fn cellRow(mat: *const Matrix(usize), ic: usize, n: usize) []const usize {
    return mat.data[ic * mat.stride ..][0..n];
}

/// Checks that hold for any valid mesh, whatever its shape.
fn expectConnInvariants(geo: *const Geo) !void {
    const t = std.testing;

    try t.expectEqual(geo.n_faces, geo.n_int_faces + geo.n_bnd_faces);
    try t.expectEqual(geo.n_int_faces, geo.int_faces.items.len);
    try t.expectEqual(geo.n_bnd_faces, geo.bnd_faces.items.len);

    // Each face is claimed by exactly as many cells as its type implies, and
    // f2c agrees with c2f in both directions.
    var n_slots: usize = 0;
    for (0..geo.n_eles) |ic| {
        for (0..geo.c2nf.items[ic]) |j| {
            const ff = geo.c2f.get(ic, j);
            if (ff == none) continue;
            n_slots += 1;
            try t.expect(ff < geo.n_faces);
            try t.expect(geo.f2c.get(ff, 0) == ic or geo.f2c.get(ff, 1) == ic);

            // c2b mirrors face_type, and c2c is set iff the face is internal
            const on_bnd = geo.face_type.items[ff] != .internal;
            try t.expectEqual(@intFromBool(on_bnd), geo.c2b.get(ic, j));
            try t.expectEqual(!on_bnd, geo.c2c.get(ic, j) != none);
        }
    }
    try t.expectEqual(2 * geo.n_int_faces + geo.n_bnd_faces, n_slots);

    for (geo.int_faces.items) |ff| {
        try t.expectEqual(FaceType.internal, geo.face_type.items[ff]);
        try t.expect(geo.f2c.get(ff, 0) != none and geo.f2c.get(ff, 1) != none);
    }
    for (geo.bnd_faces.items) |ff| {
        try t.expectEqual(FaceType.boundary, geo.face_type.items[ff]);
        try t.expect(geo.f2c.get(ff, 0) != none and geo.f2c.get(ff, 1) == none);
    }

    // c2c is symmetric: if A sees B across a face, B sees A across one too.
    for (0..geo.n_eles) |ic| {
        for (0..geo.c2nf.items[ic]) |j| {
            const nb = geo.c2c.get(ic, j);
            if (nb == none) continue;
            try t.expect(nb != ic);
            const back = cellRow(&geo.c2c, nb, geo.c2nf.items[nb]);
            try t.expect(std.mem.indexOfScalar(usize, back, ic) != null);
        }
    }

    // f2v rows are sorted ascending; face identity relies on it
    for (0..geo.n_faces) |ff| {
        for (1..geo.f2nv.items[ff]) |j| {
            try t.expect(geo.f2v.get(ff, j - 1) < geo.f2v.get(ff, j));
        }
    }

    // bc_faces row counts agree with n_faces_per_bnd and with bc_id
    for (0..geo.n_bounds) |bnd| {
        try t.expectEqual(geo.n_faces_per_bnd.items[bnd], geo.bc_faces.items[bnd].rows);
        var n: usize = 0;
        for (geo.bc_id.items) |id| {
            if (id == bnd) n += 1;
        }
        try t.expectEqual(n, geo.n_faces_per_bnd.items[bnd]);
    }
}

test "processConnectivity: 2D, 2x2 quads" {
    const t = std.testing;
    const gpa = t.allocator;

    var config: cfg.Config = undefined;
    var geo = testCreateGeo(gpa, &config, 2, .{
        .nx = 2,
        .ny = 2,
        .bc_bottom = .slip_wall,
        .bc_right = .sup_out,
        .bc_top = .symmetry,
        .bc_left = .sup_in,
    });
    defer geo.deinit();

    try geo.createMesh();
    try geo.processConnectivity();

    // 12 edges: 3 rows of 2 horizontal + 3 columns of 2 vertical. The four
    // touching the center vertex are interior; the 8 perimeter ones are not.
    try t.expectEqual(@as(usize, 12), geo.n_faces);
    try t.expectEqual(@as(usize, 4), geo.n_int_faces);
    try t.expectEqual(@as(usize, 8), geo.n_bnd_faces);
    try t.expectEqual(@as(usize, 0), geo.n_mpi_faces);

    // Face IDs follow first-encounter order over cells and their edge slots
    try t.expectEqualSlices(usize, &.{ 0, 1, 2, 3 }, cellRow(&geo.c2f, 0, 4));
    try t.expectEqualSlices(usize, &.{ 2, 4, 5, 6 }, cellRow(&geo.c2f, 1, 4));
    try t.expectEqualSlices(usize, &.{ 7, 8, 9, 1 }, cellRow(&geo.c2f, 2, 4));
    try t.expectEqualSlices(usize, &.{ 9, 10, 11, 4 }, cellRow(&geo.c2f, 3, 4));

    try t.expectEqualSlices(usize, &.{ 1, 2, 4, 9 }, geo.int_faces.items);
    try t.expectEqualSlices(usize, &.{ 0, 3, 5, 6, 7, 8, 10, 11 }, geo.bnd_faces.items);

    // f2v holds sorted vertex pairs
    try t.expectEqual(@as(usize, 0), geo.f2v.get(0, 0));
    try t.expectEqual(@as(usize, 1), geo.f2v.get(0, 1));
    try t.expectEqual(@as(usize, 1), geo.f2v.get(1, 0));
    try t.expectEqual(@as(usize, 4), geo.f2v.get(1, 1));

    // Cell 0 is the lower-left quad: bottom and left edges are on boundaries,
    // the other two are shared with cells 2 (+x) and 1 (+y).
    try t.expectEqualSlices(usize, &.{ 1, 0, 0, 1 }, cellRow(&geo.c2b, 0, 4));
    try t.expectEqualSlices(usize, &.{ none, 2, 1, none }, cellRow(&geo.c2c, 0, 4));
    try t.expectEqualSlices(usize, &.{ 9, 10, 11, 4 }, cellRow(&geo.c2f, 3, 4));

    try t.expectEqual(@as(usize, 0), geo.f2c.get(1, 0));
    try t.expectEqual(@as(usize, 2), geo.f2c.get(1, 1));
    try t.expectEqual(@as(usize, 0), geo.f2c.get(0, 0));
    try t.expectEqual(none, geo.f2c.get(0, 1));

    // bc_list is [sup_in, sup_out, slip_wall, symmetry] = [left, right, bottom, top]
    try t.expectEqualSlices(usize, &.{ 2, 0, 3, 0, 2, 1, 1, 3 }, geo.bc_id.items);
    try t.expectEqualSlices(cfg.BoundaryCondition, &.{
        .slip_wall, .sup_in,  .symmetry, .sup_in,
        .slip_wall, .sup_out, .sup_out,  .symmetry,
    }, geo.bc_type.items);

    // Two edges per side, matching what createMesh counted
    try t.expectEqualSlices(usize, &.{ 2, 2, 2, 2 }, geo.n_faces_per_bnd.items);

    try expectConnInvariants(&geo);
}

test "processConnectivity: 3D, single hex" {
    const t = std.testing;
    const gpa = t.allocator;

    var config: cfg.Config = undefined;
    var geo = testCreateGeo(gpa, &config, 3, .{
        .nx = 1,
        .ny = 1,
        .nz = 1,
        .bc_bottom = .slip_wall,
        .bc_right = .sup_out,
        .bc_top = .symmetry,
        .bc_left = .sup_in,
        .bc_front = .characteristic,
        .bc_back = .adiabatic_noslip,
    });
    defer geo.deinit();

    try geo.createMesh();
    try geo.processConnectivity();

    try t.expectEqual(@as(usize, 6), geo.n_faces);
    try t.expectEqual(@as(usize, 0), geo.n_int_faces);
    try t.expectEqual(@as(usize, 6), geo.n_bnd_faces);
    for (geo.f2nv.items) |fnv| try t.expectEqual(@as(usize, 4), fnv);

    // A cube has 12 edges. Flurry-cpp paired up each face's *sorted* vertex
    // list, which yields diagonals instead of edges and would not give 12.
    try t.expectEqual(@as(usize, 12), geo.n_edges);
    for (0..geo.n_edges) |ie| {
        try t.expect(geo.e2v.get(ie, 0) < geo.e2v.get(ie, 1));
    }

    // The hex local face table's sides line up with createMesh's 3D boundary
    // naming: bottom/top are z, left/right are x, back/front are y.
    // bc_list = [characteristic, sup_in, sup_out, slip_wall, adiabatic_noslip, symmetry]
    try t.expectEqualSlices(usize, &.{ 3, 5, 1, 2, 4, 0 }, geo.bc_id.items);
    try t.expectEqualSlices(usize, &.{ 1, 1, 1, 1, 1, 1 }, geo.n_faces_per_bnd.items);

    // Every face is on a boundary, so the cell has no neighbours
    try t.expectEqualSlices(usize, &.{ 1, 1, 1, 1, 1, 1 }, cellRow(&geo.c2b, 0, 6));
    try t.expectEqualSlices(usize, &.{ none, none, none, none, none, none }, cellRow(&geo.c2c, 0, 6));

    try expectConnInvariants(&geo);
}

test "processConnectivity: 3D, two hexes share one face" {
    const t = std.testing;
    const gpa = t.allocator;

    var config: cfg.Config = undefined;
    var geo = testCreateGeo(gpa, &config, 3, .{
        .nx = 2,
        .ny = 1,
        .nz = 1,
        .bc_bottom = .slip_wall,
        .bc_right = .sup_out,
        .bc_top = .symmetry,
        .bc_left = .sup_in,
        .bc_front = .characteristic,
        .bc_back = .adiabatic_noslip,
    });
    defer geo.deinit();

    try geo.createMesh();
    try geo.processConnectivity();

    // 2 * 6 faces with one shared => 11 distinct, 1 of them interior
    try t.expectEqual(@as(usize, 11), geo.n_faces);
    try t.expectEqual(@as(usize, 1), geo.n_int_faces);
    try t.expectEqual(@as(usize, 10), geo.n_bnd_faces);

    // The two cells are each other's only neighbour
    const shared = geo.int_faces.items[0];
    try t.expectEqual(@as(usize, 0), geo.f2c.get(shared, 0));
    try t.expectEqual(@as(usize, 1), geo.f2c.get(shared, 1));

    var n_links: usize = 0;
    for (0..2) |ic| {
        for (cellRow(&geo.c2c, ic, 6)) |nb| {
            if (nb != none) n_links += 1;
        }
    }
    try t.expectEqual(@as(usize, 2), n_links);

    try expectConnInvariants(&geo);
}

test "processConnectivity: invariants hold on larger meshes" {
    const gpa = std.testing.allocator;

    {
        var config: cfg.Config = undefined;
        var geo = testCreateGeo(gpa, &config, 2, .{
            .nx = 5,
            .ny = 4,
            .bc_bottom = .slip_wall,
            .bc_top = .symmetry,
            .bc_left = .sup_in,
            .bc_right = .sup_out,
        });
        defer geo.deinit();
        try geo.createMesh();
        try geo.processConnectivity();

        // 6 columns of 4 vertical edges + 5 rows... i.e. (nx+1)*ny + nx*(ny+1)
        try std.testing.expectEqual(@as(usize, 6 * 4 + 5 * 5), geo.n_faces);
        try std.testing.expectEqual(@as(usize, 2 * (5 + 4)), geo.n_bnd_faces);
        try expectConnInvariants(&geo);
    }

    {
        var config: cfg.Config = undefined;
        var geo = testCreateGeo(gpa, &config, 3, .{
            .nx = 3,
            .ny = 2,
            .nz = 2,
            .bc_left = .sup_in,
            .bc_right = .sup_out,
            .bc_bottom = .slip_wall,
            .bc_top = .symmetry,
            .bc_front = .characteristic,
            .bc_back = .adiabatic_noslip,
        });
        defer geo.deinit();
        try geo.createMesh();
        try geo.processConnectivity();

        // Faces normal to each axis: (n+1) planes of the other two counts
        const nx = 3;
        const ny = 2;
        const nz = 2;
        try std.testing.expectEqual(
            @as(usize, (nx + 1) * ny * nz + nx * (ny + 1) * nz + nx * ny * (nz + 1)),
            geo.n_faces,
        );
        try std.testing.expectEqual(
            @as(usize, 2 * (nx * ny + nx * nz + ny * nz)),
            geo.n_bnd_faces,
        );
        // A structured box has one edge per grid line segment
        try std.testing.expectEqual(
            @as(usize, nx * (ny + 1) * (nz + 1) + (nx + 1) * ny * (nz + 1) + (nx + 1) * (ny + 1) * nz),
            geo.n_edges,
        );
        try expectConnInvariants(&geo);
    }
}

test "processConnectivity: boundary faces with no matching boundary" {
    const t = std.testing;
    const gpa = t.allocator;

    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();

    var config: cfg.Config = undefined;
    var geo = try testGeo(gpa, arena.allocator(), &config);
    defer geo.deinit();

    // mesh_v2 is a single quad, but only two of its four edges belong to a
    // declared boundary ("Wall" = {0,1} and "Inlet" = {2,3}).
    try geo.parseGmsh(arena.allocator(), mesh_v2);
    try geo.processConnectivity();

    try t.expectEqual(@as(usize, 4), geo.n_faces);
    try t.expectEqual(@as(usize, 0), geo.n_int_faces);
    try t.expectEqual(@as(usize, 4), geo.n_bnd_faces);

    // Faces are {0,1}, {1,2}, {2,3}, {0,3}; the two diagonal-adjacent ones
    // span both boundaries and so match neither.
    try t.expectEqualSlices(usize, &.{ 0, none, 1, none }, geo.bc_id.items);
    try t.expectEqualSlices(cfg.BoundaryCondition, &.{
        .slip_wall, .none, .sup_in, .none,
    }, geo.bc_type.items);
    try t.expectEqualSlices(usize, &.{ 1, 1 }, geo.n_faces_per_bnd.items);

    try expectConnInvariants(&geo);
}

test "processConnectivity: rejects a mesh dimension mismatch" {
    const gpa = std.testing.allocator;

    var config: cfg.Config = undefined;
    var geo = testCreateGeo(gpa, &config, 2, .{ .nx = 2, .ny = 2 });
    defer geo.deinit();

    try geo.createMesh();

    // 2D quads cannot be processed as a 3D mesh, and vice versa
    geo.n_dims = 3;
    try std.testing.expectError(error.UnsupportedCellType, geo.processConnectivity());

    geo.n_dims = 4;
    try std.testing.expectError(error.UnsupportedDimension, geo.processConnectivity());
}

const std = @import("std");

const Matrix = @import("util/matrix.zig").Matrix;
const Array3 = @import("util/array3.zig").Array3;
const Array4 = @import("util/array4.zig").Array4;
const cfg = @import("config.zig");
const points = @import("points.zig");
const poly = @import("math/polynomials.zig");

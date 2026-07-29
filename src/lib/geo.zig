/// Geometry loading and processing
pub const Geo = struct {
    gpa: std.mem.Allocator,
    io: std.Io,

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

        // if (grid_rank == 0)
        //   std::cout << "Geo: Reading mesh file " << file_name << std::endl;

        // --- Read Boundary Conditions & Fluid Field(s) ---

        // Move cursor to $PhysicalNames
        var iter = std.mem.tokenizeAny(u8, mesh_file, "\n");
        while (iter.next()) |line| {
            if (std.mem.indexOf(u8, line, "$PhysicalNames")) |_| {
                break;
            }
        }
        if (iter.peek() == null) {
            @panic("$PhysicalNames tag not found in Gmsh file!");
        }

        // // Read number of boundaries and fields defined
        // mesh_file >> n_gmsh_bnds;
        // getline(mesh_file, str); // clear rest of line

        // n_bounds = 0;
        // for (int i = 0; i < n_gmsh_bnds; i++) {
        //   std::string bc_str, bc_name;
        //   std::stringstream ss;
        //   int bcdim, bcid;

        //   getline(mesh_file, str);
        //   ss << str;
        //   ss >> bcdim >> bcid >> bc_str;

        //   // Remove quotation marks from around boundary condition
        //   size_t ind = bc_str.find("\"");
        //   while (ind != std::string::npos) {
        //     bc_str.erase(ind, 1);
        //     ind = bc_str.find("\"");
        //   }
        //   bc_name = bc_str;

        //   // Convert to lowercase to match Flurry's boundary condition strings
        //   std::transform(bc_str.begin(), bc_str.end(), bc_str.begin(), ::tolower);

        //   // First, map mesh boundary to boundary condition in input file
        //   if (!params_->mesh_bounds.count(bc_str)) {
        //     std::string err_s = "Unrecognized mesh boundary: \"" + bc_str + "\"\n";
        //     err_s += "Boundary names in input file must match those in mesh file.";
        //     FatalError(err_s.c_str());
        //   }

        //   // Map the Gmsh PhysicalName to the input-file-specified Flurry boundary condition
        //   bc_str = params_->mesh_bounds[bc_str];

        //   // Next, check that the requested boundary condition exists
        //   if (!bc_str2_num.count(bc_str)) {
        //     std::string const err_s = "Unrecognized boundary condition: \"" + bc_str + "\"";
        //     FatalError(err_s.c_str());
        //   }

        //   if (bc_str.compare("fluid") == 0) {
        //     n_dims = bcdim;
        //     params_->n_dims = bcdim;
        //     bc_id_map_[bcid] = -1;
        //   } else {
        //     bc_list.push_back(bc_str2_num[bc_str]);
        //     bc_names.push_back(bc_name);
        //     bc_id_map_[bcid] = n_bounds; // Map Gmsh bcid to Flurry bound index
        //     n_bounds++;
        //   }
        // }

        // // --- Read Mesh Vertex Locations ---

        // // Move cursor to $Nodes
        // mesh_file.clear();
        // mesh_file.seekg(0, std::ios::beg);
        // while (1) {
        //   getline(mesh_file, str);
        //   if (str.find("$Nodes") != std::string::npos)
        //     break;
        //   if (mesh_file.eof())
        //     FatalError("$Nodes tag not found in Gmsh file!");
        // }

        // uint iv;
        // mesh_file >> n_verts;
        // xv.setup(n_verts, n_dims);
        // getline(mesh_file, str); // Clear end of line, just in case

        // for (0..n_verts) |i| {
        //   mesh_file >> iv >> xv(i, 0) >> xv(i, 1);
        //   if (n_dims == 3)
        //     mesh_file >> xv(i, 2);
        //   getline(mesh_file, str);
        // }

        // // --- Read Element Connectivity ---

        // // Move cursor to $Elements
        // mesh_file.clear();
        // mesh_file.seekg(0, std::ios::beg);
        // while (1) {
        //   getline(mesh_file, str);
        //   if (str.find("$Elements") != std::string::npos)
        //     break;
        //   if (mesh_file.eof())
        //     FatalError("$Elements tag not found in Gmsh file!");
        // }

        // int n_eles_gmsh;
        // std::vector<int> c2v_tmp(27, 0); // Maximum number of nodes/element possible
        // std::vector<std::set<int>> bound_points(n_bounds);
        // //  bndPtsGmsh.resize(nGmshBnds);
        // std::map<int, int> e_type2nv;
        // e_type2nv[3] = 4;  // Linear quad
        // e_type2nv[16] = 4; // Quadratic serendipity quad
        // e_type2nv[10] = 4; // Quadratic Lagrange quad
        // e_type2nv[8] = 8;  // Linear hex

        // n_bnd_pts.resize(n_bounds);

        // // Read total number of interior + boundary elements
        // mesh_file >> n_eles_gmsh;
        // getline(mesh_file, str); // Clear end of line, just in case

        // // For Gmsh node ordering, see: http://geuz.org/gmsh/doc/texinfo/gmsh.html#Node-ordering
        // int ic = 0;
        // for (int k = 0; k < n_eles_gmsh; k++) {
        //   int id, e_type, n_tags, bcid, tmp;
        //   mesh_file >> id >> e_type >> n_tags;
        //   mesh_file >> bcid;
        //   bcid = bc_id_map_[bcid];
        //   for (int tag = 0; tag < n_tags - 1; tag++)
        //     mesh_file >> tmp;

        //   if (bcid == -1) {
        //     // NOTE: Currently, only quads are supported
        //     switch (e_type) {
        //     case 2:
        //       // linear triangle
        //       c2nv.push_back(3);
        //       c2nf.push_back(3);
        //       ctype.push_back(TRI);
        //       mesh_file >> c2v_tmp[0] >> c2v_tmp[1] >> c2v_tmp[2];
        //       break;

        //     case 9:
        //       // quadratic triangle -> quadratic quad  [corner nodes, then edge-center nodes]
        //       c2nv.push_back(8);
        //       c2nf.push_back(4);
        //       ctype.push_back(QUAD);
        //       mesh_file >> c2v_tmp[0] >> c2v_tmp[1] >> c2v_tmp[2] >> c2v_tmp[4] >> c2v_tmp[5] >>
        //         c2v_tmp[7];
        //       c2v_tmp[3] = c2v_tmp[2];
        //       c2v_tmp[6] = c2v_tmp[2];
        //       break;

        //     case 3:
        //       // linear quadrangle
        //       c2nv.push_back(4);
        //       c2nf.push_back(4);
        //       ctype.push_back(QUAD);
        //       mesh_file >> c2v_tmp[0] >> c2v_tmp[1] >> c2v_tmp[2] >> c2v_tmp[3];
        //       break;

        //     case 16:
        //       // quadratic 8-node (serendipity) quadrangle
        //       c2nv.push_back(8);
        //       c2nf.push_back(4);
        //       ctype.push_back(QUAD);
        //       mesh_file >> c2v_tmp[0] >> c2v_tmp[1] >> c2v_tmp[2] >> c2v_tmp[3] >> c2v_tmp[4] >>
        //         c2v_tmp[5] >> c2v_tmp[6] >> c2v_tmp[7];
        //       break;

        //     case 10:
        //       // quadratic (9-node Lagrange) quadrangle (read as 8-node serendipity)
        //       c2nv.push_back(9);
        //       c2nf.push_back(4);
        //       ctype.push_back(QUAD);
        //       for (int i = 0; i < 9; i++)
        //         mesh_file >> c2v_tmp[i];
        //       break;

        //     case 36:
        //       // cubic (16-node Lagrange) quadrangle
        //       c2nv.push_back(16);
        //       c2nf.push_back(4);
        //       ctype.push_back(QUAD);
        //       for (int i = 0; i < 16; i++)
        //         mesh_file >> c2v_tmp[i];
        //       break;

        //     case 37:
        //       // quartic (25-node Lagrange) quadrangle
        //       c2nv.push_back(25);
        //       c2nf.push_back(4);
        //       ctype.push_back(QUAD);
        //       for (int i = 0; i < 25; i++)
        //         mesh_file >> c2v_tmp[i];
        //       break;

        //     case 38:
        //       // quintic (36-node Lagrange) quadrangle
        //       c2nv.push_back(36);
        //       c2v_tmp.resize(36);
        //       c2nf.push_back(4);
        //       ctype.push_back(QUAD);
        //       for (int i = 0; i < 36; i++)
        //         mesh_file >> c2v_tmp[i];
        //       break;

        //     case 47:
        //       // 6th-order 49-node Lagrange quadrangle
        //       c2nv.push_back(49);
        //       c2v_tmp.resize(49);
        //       c2nf.push_back(4);
        //       ctype.push_back(QUAD);
        //       for (int i = 0; i < 49; i++)
        //         mesh_file >> c2v_tmp[i];
        //       break;

        //     case 48:
        //       // 7th-order 64-node Lagrange quadrangle
        //       c2nv.push_back(64);
        //       c2v_tmp.resize(64);
        //       c2nf.push_back(4);
        //       ctype.push_back(QUAD);
        //       for (int i = 0; i < 64; i++)
        //         mesh_file >> c2v_tmp[i];
        //       break;

        //     case 49:
        //       // 8th-order 81-node Lagrange quadrangle
        //       c2nv.push_back(81);
        //       c2v_tmp.resize(81);
        //       c2nf.push_back(4);
        //       ctype.push_back(QUAD);
        //       for (int i = 0; i < 81; i++)
        //         mesh_file >> c2v_tmp[i];
        //       break;

        //     case 50:
        //       // 9th-order 100-node Lagrange quadrangle
        //       c2nv.push_back(100);
        //       c2v_tmp.resize(100);
        //       c2nf.push_back(4);
        //       ctype.push_back(QUAD);
        //       for (int i = 0; i < 100; i++)
        //         mesh_file >> c2v_tmp[i];
        //       break;

        //     case 51:
        //       // 10th-order 121-node Lagrange quadrangle
        //       c2nv.push_back(121);
        //       c2v_tmp.resize(121);
        //       c2nf.push_back(4);
        //       ctype.push_back(QUAD);
        //       for (int i = 0; i < 121; i++)
        //         mesh_file >> c2v_tmp[i];
        //       break;

        //     case 5:
        //       // Linear hexahedron
        //       c2nv.push_back(8);
        //       c2nf.push_back(6);
        //       ctype.push_back(HEX);
        //       for (int i = 0; i < 8; i++)
        //         mesh_file >> c2v_tmp[i];
        //       break;

        //     case 17:
        //       // Quadratic (20-Node Serendipity) Hexahedron
        //       c2nv.push_back(20);
        //       c2nf.push_back(6);
        //       ctype.push_back(HEX);
        //       // Corner Nodes
        //       mesh_file >> c2v_tmp[0] >> c2v_tmp[1] >> c2v_tmp[2] >> c2v_tmp[3] >> c2v_tmp[4] >>
        //         c2v_tmp[5] >> c2v_tmp[6] >> c2v_tmp[7];
        //       // Edge Nodes
        //       mesh_file >> c2v_tmp[8] >> c2v_tmp[11] >> c2v_tmp[12] >> c2v_tmp[9] >> c2v_tmp[13] >>
        //         c2v_tmp[10];
        //       mesh_file >> c2v_tmp[14] >> c2v_tmp[15] >> c2v_tmp[16] >> c2v_tmp[19] >> c2v_tmp[17] >>
        //         c2v_tmp[18];
        //       break;

        //     case 12:
        //       // Quadratic (27-Node Lagrange) Hexahedron
        //       c2nv.push_back(27);
        //       c2nf.push_back(6);
        //       ctype.push_back(HEX);
        //       c2v_tmp.resize(27);
        //       for (int i = 0; i < c2nv.back(); i++)
        //         mesh_file >> c2v_tmp[i];
        //       break;

        //     case 92:
        //       // Cubic Hexahedron
        //       c2nv.push_back(64);
        //       c2nf.push_back(6);
        //       ctype.push_back(HEX);
        //       c2v_tmp.resize(64);
        //       for (int i = 0; i < c2nv.back(); i++)
        //         mesh_file >> c2v_tmp[i];
        //       break;

        //     case 93:
        //       // Quartic Hexahedron
        //       c2nv.push_back(125);
        //       c2nf.push_back(6);
        //       ctype.push_back(HEX);
        //       c2v_tmp.resize(125);
        //       for (int i = 0; i < c2nv.back(); i++)
        //         mesh_file >> c2v_tmp[i];
        //       break;

        //     case 94:
        //       // Quintic Hexahedron
        //       c2nv.push_back(216);
        //       c2nf.push_back(6);
        //       ctype.push_back(HEX);
        //       c2v_tmp.resize(216);
        //       for (int i = 0; i < c2nv.back(); i++)
        //         mesh_file >> c2v_tmp[i];
        //       break;

        //     case 4:
        //       // Linear tetrahedron; read as collapsed-face hex
        //       c2nv.push_back(4);
        //       c2nf.push_back(4);
        //       ctype.push_back(HEX);
        //       mesh_file >> c2v_tmp[0] >> c2v_tmp[1] >> c2v_tmp[2] >> c2v_tmp[4];
        //       c2v_tmp[3] = 2;
        //       c2v_tmp[5] = c2v_tmp[4];
        //       c2v_tmp[6] = c2v_tmp[4];
        //       c2v_tmp[6] = c2v_tmp[4];
        //       break;

        //     case 6:
        //       // Linear prism; read as collapsed-face hex
        //       c2nv.push_back(8);
        //       c2nf.push_back(6);
        //       ctype.push_back(HEX);
        //       mesh_file >> c2v_tmp[0] >> c2v_tmp[1] >> c2v_tmp[2] >> c2v_tmp[4] >> c2v_tmp[5] >>
        //         c2v_tmp[6];
        //       c2v_tmp[3] = c2v_tmp[2];
        //       c2v_tmp[7] = c2v_tmp[6];
        //       break;

        //     default:
        //       std::cout << "Gmsh element ID " << k << ", Gmsh Element Type = " << e_type << std::endl;
        //       FatalError("element type not recognized");
        //       break;
        //     }

        //     // Increase the size of c2v (max # of vertices per cell) if needed
        //     if (c2v.getDim1() < (uint)c2nv[ic]) {
        //       for (int dim = c2v.getDim1(); dim < c2nv[ic]; dim++) {
        //         c2v.addCol();
        //       }
        //     }

        //     // Number of nodes in c2v_tmp may vary, so use pointer rather than vector
        //     c2v.insertRow(c2v_tmp.data(), -1, c2nv[ic]);

        //     // Shift every value of c2v by -1 (Gmsh is 1-indexed; we need 0-indexed)
        //     for (int k = 0; k < c2nv[ic]; k++) {
        //       if (c2v(ic, k) != 0) {
        //         c2v(ic, k)--;
        //       }
        //     }

        //     ic++;
        //     getline(mesh_file, str); // skip end of line
        //   } else {
        //     // Boundary cell; put vertices into bndPts
        //     int n_pts_face = 0;
        //     switch (e_type) {
        //     case 1: // Linear edge
        //       n_pts_face = 2;
        //       break;

        //     case 2: // Linear triangle
        //       n_pts_face = 3;
        //       break;

        //     case 3:  // Linear quad
        //     case 10: // Quadratic (Lagrange) quad
        //     case 16: // Quadratic (Serendipity) quad
        //     case 36: // Cubic quad
        //     case 37: // Quartic quad
        //     case 38: // Quintic quad
        //       n_pts_face = 4;
        //       break;

        //     case 8: // Quadratic edge
        //       n_pts_face = 3;
        //       break;

        //     case 26: // Cubic Edge
        //       n_pts_face = 4;
        //       break;

        //     case 27: // Quartic Edge
        //       n_pts_face = 5;
        //       break;

        //     case 28: // Quintic Edge
        //       n_pts_face = 6;
        //       break;

        //     case 62: // Order 6
        //       n_pts_face = 7;
        //       break;

        //     case 63: // Order 7
        //       n_pts_face = 8;
        //       break;

        //     case 64: // Order 8
        //       n_pts_face = 9;
        //       break;

        //     case 65: // Order 9
        //       n_pts_face = 10;
        //       break;

        //     case 66: // Order 10
        //       n_pts_face = 11;
        //       break;

        //     default:
        //       std::cout << "Gmsh element ID " << k << ", Gmsh Element Type = " << e_type << std::endl;
        //       FatalError("Boundary Element (Face) Type Not Recognized!");
        //     }

        //     for (int i = 0; i < n_pts_face; i++) {
        //       mesh_file >> iv;
        //       iv--;
        //       bound_points[bcid].insert(iv);
        //       // bndPtsGmsh[gmshID].push_back(iv);
        //     }
        //     getline(mesh_file, str);
        //   }
        // } // End of loop over entities

        // //  for (int i = 0; i < nGmshBnds; i++) {
        // //    std::sort(bndPtsGmsh[i].begin(),bndPtsGmsh[i].end());
        // //    bndPtsGmsh[i].erase( std::unique(bndPtsGmsh[i].begin(),bndPtsGmsh[i].end()),
        // //    bndPtsGmsh[i].end() );
        // //  }

        // int max_n_bnd_pts = 0;
        // for (int i = 0; i < n_bounds; i++) {
        //   n_bnd_pts[i] = bound_points[i].size();
        //   max_n_bnd_pts = std::max(max_n_bnd_pts, n_bnd_pts[i]);
        // }

        // // Copy temp boundPoints data into bndPts matrix
        // bnd_pts.setup(n_bounds, max_n_bnd_pts);
        // for (int i = 0; i < n_bounds; i++) {
        //   int j = 0;
        //   for (auto& it : bound_points[i]) {
        //     bnd_pts(i, j) = it;
        //     j++;
        //   }
        // }

        // n_eles = c2v.getDim0();

        // mesh_file.close();
    }

    pub fn createMesh(geo: *Geo) !void {
        _ = geo; // autofix
    }
};

const std = @import("std");

const Matrix = @import("../matrix.zig").Matrix;
const Array3 = @import("../array3.zig").Array3;
const Array4 = @import("../array4.zig").Array4;
const points = @import("../points.zig");
const poly = @import("../math/polynomials.zig");

const std = @import("std");
const Io = std.Io;
const flurry = @import("flurry");
const config = flurry.config;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const alloc = init.gpa;

    const cwd = Io.Dir.cwd();
    const cfg_file = "samples/cylinder.cfg.ziggy";

    // 1. parse (arena-backed — pc.deinit() frees everything at once)
    var pc = config.loader.parse(io, alloc, cwd, cfg_file) catch |err| {
        std.debug.print("Failed to parse {s}: {}\n", .{ cfg_file, err });
        return err;
    };
    defer pc.deinit();

    // 2. post-parse init (derive computed values)
    config.loader.initialize(&pc.value);

    std.debug.print("Loaded: n_dims={}, mesh={s}, order={d}\n", .{
        pc.value.core.n_dims,
        pc.value.core.mesh_file,
        pc.value.core.order,
    });

    var bc_it = pc.value.boundary_conditions.mesh_bounds.fields.iterator();
    while (bc_it.next()) |entry| {
        std.debug.print("  boundary '{s}' -> {s}\n", .{ entry.key_ptr.*, @tagName(entry.value_ptr.*) });
    }
}

const std = @import("std");
const Io = std.Io;
const flurry = @import("flurry");
const config = flurry.config;
const gpu = flurry.gpu;
const Run = flurry.driver.Run;

const usage =
    \\usage: flurry [--gpu] <case.cfg.ziggy>
    \\
    \\  --gpu   run the ported operators on a Vulkan compute device
    \\
;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;

    var stderr_buf: [1024]u8 = undefined;
    var stderr = Io.File.stderr().writer(io, &stderr_buf);

    var args = std.process.Args.Iterator.init(init.minimal.args);
    defer args.deinit();
    _ = args.skip();

    var cfg_file: ?[]const u8 = null;
    var use_gpu = false;
    while (args.next()) |a| {
        if (std.mem.eql(u8, a, "--gpu")) use_gpu = true else cfg_file = a;
    }

    const path = cfg_file orelse {
        try stderr.interface.writeAll(usage);
        try stderr.interface.flush();
        return error.MissingArgument;
    };

    // The arena behind `pc` owns the config's strings, and both the mesh and the
    // solver keep references into them -- so it has to outlive the run.
    var pc = config.Loader.parse(io, gpa, Io.Dir.cwd(), path) catch |err| {
        try stderr.interface.print("failed to read {s}: {t}\n", .{ path, err });
        try stderr.interface.flush();
        return err;
    };
    defer pc.deinit();

    config.Loader.initialize(&pc.value);

    var stdout_buf: [4096]u8 = undefined;
    var stdout = Io.File.stdout().writer(io, &stdout_buf);
    // A run that fails partway has usually printed the reports that explain why
    defer stdout.interface.flush() catch {};

    // Declared before the run so it outlives the solver that borrows it.
    var device: ?gpu.Device = null;
    defer if (device) |*d| d.deinit();
    if (use_gpu) {
        device = gpu.Device.init(gpa, .{}) catch |err| {
            try stderr.interface.print("--gpu: no compute device ({t})\n", .{err});
            try stderr.interface.flush();
            return err;
        };
    }

    var run: Run = undefined;
    run.init(gpa, io, &pc.value) catch |err| {
        try stderr.interface.print("setup failed: {t}\n", .{err});
        try stderr.interface.flush();
        return err;
    };
    defer run.deinit();

    if (device) |*d| {
        run.solver.device = d;
        try stdout.interface.print("\n device    {s}\n", .{d.name()});
    }

    try run.run(&stdout.interface);
}

const std = @import("std");
const Io = std.Io;
const flurry = @import("flurry");
const config = flurry.config;
const Run = flurry.driver.Run;

const usage =
    \\usage: flurry <case.cfg.ziggy>
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

    const cfg_file = args.next() orelse {
        try stderr.interface.writeAll(usage);
        try stderr.interface.flush();
        return error.MissingArgument;
    };

    // The arena behind `pc` owns the config's strings, and both the mesh and the
    // solver keep references into them -- so it has to outlive the run.
    var pc = config.Loader.parse(io, gpa, Io.Dir.cwd(), cfg_file) catch |err| {
        try stderr.interface.print("failed to read {s}: {t}\n", .{ cfg_file, err });
        try stderr.interface.flush();
        return err;
    };
    defer pc.deinit();

    config.Loader.initialize(&pc.value);

    var stdout_buf: [4096]u8 = undefined;
    var stdout = Io.File.stdout().writer(io, &stdout_buf);
    // A run that fails partway has usually printed the reports that explain why
    defer stdout.interface.flush() catch {};

    var run: Run = undefined;
    run.init(gpa, io, &pc.value) catch |err| {
        try stderr.interface.print("setup failed: {t}\n", .{err});
        try stderr.interface.flush();
        return err;
    };
    defer run.deinit();

    try run.run(&stdout.interface);
}

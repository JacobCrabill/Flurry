const std = @import("std");
const Io = std.Io;

const flurry = @import("flurry");

pub fn main(init: std.process.Init) !void {
    _ = init;
    // 1. parse cli arguments
    // 2. read input file for simulation config
    // 3. perform setup shared across all processes
    // 4. (optional) configure multiprocessing environment (MPI-lite)
    // 5. (optional) fork child processes
    // 6. setup GPU (Spock) context
    // 7. setup solver
    // 8. iterate
}

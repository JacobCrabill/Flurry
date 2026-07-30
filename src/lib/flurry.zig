const std = @import("std");
const ziggy = @import("ziggy");

pub const config = @import("config.zig");
pub const geo = @import("geo.zig");
pub const element = @import("element.zig");
pub const flux = @import("flux.zig");
pub const faces = @import("faces.zig");
pub const solver = @import("solver.zig");
pub const driver = @import("driver.zig");
pub const vtu = @import("vtu.zig");
pub const testcase = @import("testcase.zig");
pub const points = @import("points.zig");
pub const polynomials = @import("math/polynomials.zig");

pub const eles = struct {
    pub const Quad = @import("eles/quads.zig").Quad;
};

test {
    std.testing.refAllDecls(@This());

    _ = @import("math/polynomials_test.zig");
    _ = @import("util/matrix.zig");
    _ = @import("util/array3.zig");
    _ = @import("util/array4.zig");
    _ = @import("config_test.zig");
    _ = @import("eles/quads_test.zig");
    _ = @import("solver_test.zig");
    _ = @import("driver_test.zig");
    _ = @import("vtu_test.zig");
    _ = @import("testcase_test.zig");
}

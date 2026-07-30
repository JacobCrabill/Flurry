const std = @import("std");
const ziggy = @import("ziggy");

pub const config = @import("config.zig");
pub const geo = @import("geo.zig");
pub const element = @import("element.zig");
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
}

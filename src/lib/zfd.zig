const std = @import("std");

pub const polynomials = @import("math/polynomials.zig");

test {
    std.testing.refAllDecls(@This());

    _ = @import("math/polynomials_test.zig");
    _ = @import("matrix.zig");
}

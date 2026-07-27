const std = @import("std");

pub const polynomials = @import("math/polynomials.zig");

test {
    std.testing.refAllDecls(@This());

    _ = @import("math/polynomials_test.zig");
    _ = @import("util/matrix.zig");
    _ = @import("util/array3.zig");
    _ = @import("util/array4.zig");
}

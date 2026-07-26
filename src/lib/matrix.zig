const std = @import("std");

/// Simple template for a 2D row-major matrix
pub fn Matrix(T: type) type {
    return struct {
        const Self = @This();
        rows: usize,
        cols: usize,
        stride: usize, // TODO: decide if useful or not
        data: []T,

        /// Initialize a new Matrix(T) with the given size and stride.
        ///
        /// If stride *not* given, defaults to `cols` (fully dense matrix).
        /// If stride *is* given, the point is to be some multiple of a cache line.
        pub fn init(gpa: std.mem.Allocator, rows: usize, cols: usize, stride_opt: ?usize) !Self {
            const stride = stride_opt orelse cols;
            std.debug.assert(stride >= cols);
            const mat: Self = .{
                .rows = rows,
                .cols = cols,
                .stride = stride,
                .data = try gpa.alloc(T, rows * stride), // TODO: allocAligned needed?
            };
            @memset(mat.data, 0.0);
            return mat;
        }

        pub fn deinit(self: *Self, gpa: std.mem.Allocator) void {
            self.rows = 0;
            self.cols = 0;
            self.stride = 0;
            gpa.free(self.data);
        }

        /// Get the element at (row, col)
        pub fn get(self: *const Self, row: usize, col: usize) T {
            std.debug.assert(row < self.rows and col < self.cols);
            return self.data[self.stride * row + col];
        }

        /// Get a pointer to the element at (row, col)
        pub fn at(self: *Self, row: usize, col: usize) *T {
            std.debug.assert(row < self.rows and col < self.cols);
            return &self.data[self.stride * row + col];
        }
    };
}

test Matrix {
    const gpa = std.testing.allocator;
    var mat = try Matrix(f64).init(gpa, 8, 8, 16);
    defer mat.deinit(gpa);

    try std.testing.expectEqual(mat.data.len, 8 * 16);
    for (mat.data) |v| try std.testing.expectEqual(0.0, v);
}

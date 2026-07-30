const std = @import("std");

pub const Array4d = Array4(f64);
pub const Array4f = Array4(f32);
pub const Array4u = Array4(u32);
pub const Array4i = Array4(i32);
pub const Array4z = Array4(usize);

/// Simple template for a 3D row-major array
pub fn Array4(T: type) type {
    return struct {
        const Self = @This();
        dims: [4]usize = .{ 0, 0, 0, 0 },
        data: []T = &.{},

        pub const empty: Self = .{};

        /// Initialize a new Array4(T) with the given size.
        /// Initializes the full array to 0.
        pub fn init(gpa: std.mem.Allocator, books: usize, pages: usize, rows: usize, cols: usize) !Self {
            const mat: Self = .{
                .dims = .{ books, pages, rows, cols },
                .data = try gpa.alloc(T, books * pages * rows * cols), // TODO: allocAligned needed?
            };
            @memset(mat.data, @as(T, 0));
            return mat;
        }

        /// Free the data and clear the dimensions
        pub fn deinit(self: *Self, gpa: std.mem.Allocator) void {
            self.dims = .{ 0, 0, 0, 0 };
            gpa.free(self.data);
        }

        /// Get the element at (row, col)
        pub fn get(self: *const Self, i: usize, j: usize, k: usize, l: usize) T {
            std.debug.assert(i < self.dims[0] and j < self.dims[1] and k < self.dims[2] and l < self.dims[3]);
            return self.data[l + self.dims[3] * (k + self.dims[2] * (j + self.dims[1] * i))];
        }

        /// Get a pointer to the element at (row, col)
        pub fn at(self: *Self, i: usize, j: usize, k: usize, l: usize) *T {
            std.debug.assert(i < self.dims[0] and j < self.dims[1] and k < self.dims[2] and l < self.dims[3]);
            return &self.data[l + self.dims[3] * (k + self.dims[2] * (j + self.dims[1] * i))];
        }
    };
}

test Array4 {
    const gpa = std.testing.allocator;
    var mat = try Array4(i64).init(gpa, 12, 4, 5, 2);
    defer mat.deinit(gpa);

    try std.testing.expectEqual(mat.data.len, 12 * 4 * 5 * 2);
    for (mat.data) |v| try std.testing.expectEqual(0, v);

    var idx: i64 = 0;
    for (0..12) |i| {
        for (0..4) |j| {
            for (0..5) |k| {
                for (0..2) |l| {
                    mat.at(i, j, k, l).* = idx;
                    idx += 1;
                }
            }
        }
    }

    idx = 0;
    for (0..12) |i| {
        for (0..4) |j| {
            for (0..5) |k| {
                for (0..2) |l| {
                    try std.testing.expectEqual(idx, mat.get(i, j, k, l));
                    try std.testing.expectEqual(idx, mat.data[@intCast(idx)]);
                    idx += 1;
                }
            }
        }
    }
}

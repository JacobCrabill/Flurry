const std = @import("std");

pub const Array3d = Array3(f64);
pub const Array3f = Array3(f32);
pub const Array3u = Array3(u32);
pub const Array3i = Array3(i32);
pub const Array3z = Array3(usize);

/// Simple template for a 3D row-major array
pub fn Array3(T: type) type {
    return struct {
        const Self = @This();
        dims: [3]usize,
        data: []T,

        /// Initialize a new Array3(T) with the given size and stride.
        /// Initializes the full array to 0.
        pub fn init(gpa: std.mem.Allocator, pages: usize, rows: usize, cols: usize) !Self {
            const mat: Self = .{
                .dims = .{ pages, rows, cols },
                .data = try gpa.alloc(T, pages * rows * cols), // TODO: allocAligned needed?
            };
            @memset(mat.data, @as(T, 0));
            return mat;
        }

        /// Free the data and clear the dimensions
        pub fn deinit(self: *Self, gpa: std.mem.Allocator) void {
            self.dims = .{ 0, 0, 0 };
            gpa.free(self.data);
        }

        /// Get the element at (row, col)
        pub fn get(self: *const Self, i: usize, j: usize, k: usize) T {
            std.debug.assert(i < self.dims[0] and j < self.dims[1] and k < self.dims[2]);
            return self.data[k + self.dims[2] * (j + self.dims[1] * i)];
        }

        /// Get a pointer to the element at (row, col)
        pub fn at(self: *Self, i: usize, j: usize, k: usize) *T {
            std.debug.assert(i < self.dims[0] and j < self.dims[1] and k < self.dims[2]);
            return &self.data[k + self.dims[2] * (j + self.dims[1] * i)];
        }
    };
}

test Array3 {
    const gpa = std.testing.allocator;
    var mat = try Array3(i64).init(gpa, 12, 4, 5);
    defer mat.deinit(gpa);

    try std.testing.expectEqual(mat.data.len, 12 * 4 * 5);
    for (mat.data) |v| try std.testing.expectEqual(0, v);

    var idx: i64 = 0;
    for (0..12) |i| {
        for (0..4) |j| {
            for (0..5) |k| {
                mat.at(i, j, k).* = idx;
                idx += 1;
            }
        }
    }

    idx = 0;
    for (0..12) |i| {
        for (0..4) |j| {
            for (0..5) |k| {
                try std.testing.expectEqual(idx, mat.get(i, j, k));
                try std.testing.expectEqual(idx, mat.data[@intCast(idx)]);
                idx += 1;
            }
        }
    }
}

//! Boundary gradients at the global flux points.
//!
//! The GPU half of `Faces.applyBcsGrad`, for the conditions that have a kernel:
//! every one of them extrapolates, so the right side of a boundary flux point
//! sees the left side's gradient. One thread per boundary flux point.
//!
//! The adiabatic wall does more than extrapolate -- it removes the wall-normal
//! temperature gradient -- but it has no `face_bcs` kernel either, so a case
//! using one keeps the whole step on the CPU and never reaches this. See
//! `Solver.applyFaceBcs`.
//!
//! Layout:
//!
//!     du  (slot, dim, var, gfpt)   gfpt fastest

comptime {
    if (@import("builtin").target.cpu.arch.isSpirV()) {
        @export(&faceBcsGrad, .{ .name = "face_bcs_grad" });
    }
}

pub const PushConstants = extern struct {
    n_gfpts: u32,
    n_gfpts_int: u32,
    n_gfpts_bnd: u32,
    n_vars: u32,
    n_dims: u32,
};

pub const WgSize = extern struct {
    pub const x: u32 = 128;
    pub const y: u32 = 1;
    pub const z: u32 = 1;
};

const F64Array = @SpirvType(.{ .runtime_array = f64 });
const F64Buf = extern struct { data: F64Array };

const du = @extern(*addrspace(.storage_buffer) F64Buf, .{
    .name = "du",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 0 } },
});

const pc = @extern(*addrspace(.push_constant) const PushConstants, .{ .name = "pc" });

fn faceBcsGrad() callconv(.{ .spirv_kernel = .{ .x = WgSize.x, .y = WgSize.y, .z = WgSize.z } }) void {
    const i = std.spirv.global_invocation_id[0];
    if (i >= pc.n_gfpts_bnd) return;

    const gf = pc.n_gfpts_int + i;
    const slot = pc.n_gfpts * pc.n_vars * pc.n_dims;

    for (0..pc.n_dims) |dim| {
        const base = gf + pc.n_gfpts * (pc.n_vars * dim);
        for (0..pc.n_vars) |n| {
            const src = base + pc.n_gfpts * n;
            du.data[src + slot] = du.data[src];
        }
    }
}

const std = @import("std");

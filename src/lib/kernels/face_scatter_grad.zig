//! Each element's flux-point solution gradient -> the shared face arrays.
//!
//! The GPU half of `Solver.scatterGradToFaces`, and the gradient's counterpart
//! to `face_scatter`. One thread per `(fpt, ele)`; two elements meet at a global
//! flux point and write different slots, so nothing needs synchronizing.
//!
//! It is a kernel of its own rather than `n_dims` dispatches of `face_scatter`
//! because the destination's two slots are `n_dims` blocks apart, not one:
//!
//!     du_fpts   (dim, fpt, var, ele)      ele fastest
//!     faces_du  (slot, dim, var, gfpt)    gfpt fastest
//!
//! No equation appears here, and the dimension is only a loop bound over data
//! being moved, so one build serves every case.

comptime {
    if (@import("builtin").target.cpu.arch.isSpirV()) {
        @export(&faceScatterGrad, .{ .name = "face_scatter_grad" });
    }
}

pub const PushConstants = extern struct {
    n_fpts: u32,
    n_eles: u32,
    n_vars: u32,
    n_gfpts: u32,
    n_dims: u32,
};

pub const WgSize = extern struct {
    pub const x: u32 = 128;
    pub const y: u32 = 1;
    pub const z: u32 = 1;
};

const F64Array = @SpirvType(.{ .runtime_array = f64 });
const U32Array = @SpirvType(.{ .runtime_array = u32 });
const F64Buf = extern struct { data: F64Array };
const U32Buf = extern struct { data: U32Array };

const du_fpts = @extern(*addrspace(.storage_buffer) const F64Buf, .{
    .name = "du_fpts",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 0 } },
});
const fpt2gfpt = @extern(*addrspace(.storage_buffer) const U32Buf, .{
    .name = "fpt2gfpt",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 1 } },
});
const fpt2slot = @extern(*addrspace(.storage_buffer) const U32Buf, .{
    .name = "fpt2slot",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 2 } },
});
const faces_du = @extern(*addrspace(.storage_buffer) F64Buf, .{
    .name = "faces_du",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 3 } },
});

/// Marks a cell face with no global flux point behind it -- a collapsed face.
pub const none: u32 = 0xFFFF_FFFF;

const pc = @extern(*addrspace(.push_constant) const PushConstants, .{ .name = "pc" });

fn faceScatterGrad() callconv(.{ .spirv_kernel = .{ .x = WgSize.x, .y = WgSize.y, .z = WgSize.z } }) void {
    const tid = std.spirv.global_invocation_id[0];
    if (tid >= pc.n_fpts * pc.n_eles) return;

    const fpt: u32 = @divTrunc(tid, pc.n_eles);
    const ele: u32 = @mod(tid, pc.n_eles);

    const map = ele + pc.n_eles * fpt;
    const gf = fpt2gfpt.data[map];
    if (gf == none) return;
    const slot = fpt2slot.data[map];

    const src_dim = pc.n_eles * pc.n_vars * pc.n_fpts;
    const dst_slot = pc.n_gfpts * pc.n_vars * pc.n_dims;

    for (0..pc.n_dims) |dim| {
        const src = ele + pc.n_eles * (pc.n_vars * fpt) + src_dim * dim;
        const dst = gf + pc.n_gfpts * (pc.n_vars * dim) + dst_slot * slot;
        for (0..pc.n_vars) |n| {
            faces_du.data[dst + pc.n_gfpts * n] = du_fpts.data[src + pc.n_eles * n];
        }
    }
}

const std = @import("std");

//! Each element's flux-point solution -> the shared face arrays.
//!
//! The GPU half of `Solver.scatterUToFaces`. One thread per `(fpt, ele)`. Two
//! elements meet at a global flux point and write different slots of it, so no
//! two threads touch the same location and there is nothing to synchronize.
//!
//! Layouts, all the solver's:
//!
//!     u_fpts     (fpt, var, ele)     ele fastest
//!     fpt2gfpt   (fpt, ele)          which global flux point, or `none`
//!     fpt2slot   (fpt, ele)          which side of it, 0 or 1
//!     faces_u    (slot, var, gfpt)   gfpt fastest
//!
//! The index arrays are `usize` on the host; they are narrowed to `u32` on
//! upload, `none` with them, since a mesh with four billion flux points is not
//! a thing this solver will see.

comptime {
    if (@import("builtin").target.cpu.arch.isSpirV()) {
        @export(&faceScatter, .{ .name = "face_scatter" });
    }
}

/// Marks a cell face with no global flux point behind it -- a collapsed face.
/// Must match `gpu.none_u32` on the host.
pub const none: u32 = 0xFFFF_FFFF;

pub const PushConstants = extern struct {
    n_fpts: u32,
    n_eles: u32,
    n_vars: u32,
    n_gfpts: u32,
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

const u_fpts = @extern(*addrspace(.storage_buffer) const F64Buf, .{
    .name = "u_fpts",
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
const faces_u = @extern(*addrspace(.storage_buffer) F64Buf, .{
    .name = "faces_u",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 3 } },
});

const pc = @extern(*addrspace(.push_constant) const PushConstants, .{ .name = "pc" });

fn faceScatter() callconv(.{ .spirv_kernel = .{ .x = WgSize.x, .y = WgSize.y, .z = WgSize.z } }) void {
    const tid = std.spirv.global_invocation_id[0];
    if (tid >= pc.n_fpts * pc.n_eles) return;

    const fpt: u32 = @divTrunc(tid, pc.n_eles);
    const ele: u32 = @mod(tid, pc.n_eles);

    const map = ele + pc.n_eles * fpt;
    const gf = fpt2gfpt.data[map];
    if (gf == none) return;
    const slot = fpt2slot.data[map];

    const src = ele + pc.n_eles * (pc.n_vars * fpt);
    const dst = gf + pc.n_gfpts * (pc.n_vars * slot);
    for (0..pc.n_vars) |n| {
        faces_u.data[dst + pc.n_gfpts * n] = u_fpts.data[src + pc.n_eles * n];
    }
}

const std = @import("std");

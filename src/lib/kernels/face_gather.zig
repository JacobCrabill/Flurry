//! The common normal flux -> each element's own flux-point array.
//!
//! The GPU half of `Solver.gatherCommonFFromFaces`, and the exact inverse of
//! `face_scatter`: one thread per `(fpt, ele)`, each reading the slot its
//! element owns. A cell face with no global flux point behind it carries no
//! flux, so it is zeroed rather than left as it was.
//!
//! Layouts, all the solver's:
//!
//!     faces_f    (slot, var, gfpt)   gfpt fastest
//!     fpt2gfpt   (fpt, ele)          which global flux point, or `none`
//!     fpt2slot   (fpt, ele)          which side of it, 0 or 1
//!     f_comm     (fpt, var, ele)     ele fastest
//!
//! The index arrays are `usize` on the host; they are narrowed to `u32` on
//! upload, `none` with them, since a mesh with four billion flux points is not
//! a thing this solver will see.

comptime {
    if (@import("builtin").target.cpu.arch.isSpirV()) {
        @export(&faceGather, .{ .name = "face_gather" });
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

const faces_f = @extern(*addrspace(.storage_buffer) const F64Buf, .{
    .name = "faces_f",
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
const f_comm = @extern(*addrspace(.storage_buffer) F64Buf, .{
    .name = "f_comm",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 3 } },
});

const pc = @extern(*addrspace(.push_constant) const PushConstants, .{ .name = "pc" });

fn faceGather() callconv(.{ .spirv_kernel = .{ .x = WgSize.x, .y = WgSize.y, .z = WgSize.z } }) void {
    const tid = std.spirv.global_invocation_id[0];
    if (tid >= pc.n_fpts * pc.n_eles) return;

    const fpt: u32 = @divTrunc(tid, pc.n_eles);
    const ele: u32 = @mod(tid, pc.n_eles);

    const map = ele + pc.n_eles * fpt;
    const gf = fpt2gfpt.data[map];
    const dst = ele + pc.n_eles * (pc.n_vars * fpt);

    if (gf == none) {
        for (0..pc.n_vars) |n| f_comm.data[dst + pc.n_eles * n] = 0.0;
        return;
    }

    const slot = fpt2slot.data[map];
    const src = gf + pc.n_gfpts * (pc.n_vars * slot);
    for (0..pc.n_vars) |n| {
        f_comm.data[dst + pc.n_eles * n] = faces_f.data[src + pc.n_gfpts * n];
    }
}

const std = @import("std");

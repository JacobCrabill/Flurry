//! Single-valued interface solution, for the viscous gradient correction.
//!
//! The GPU half of `Faces.computeCommonU`. One thread per global flux point: it
//! blends the two sides' `u_ldg` with the LDG bias and writes the same value to
//! both slots.
//!
//! Unlike `f_comm`, this is a *value*, not a normal flux: it is fed to
//! `oppD_fpts` as the DFR interpolant's endpoint, so it carries neither the face
//! measure nor either element's outward sign.
//!
//! No equation and no dimension appear here -- it is a per-variable blend -- so
//! one build serves every case.
//!
//! Layouts:
//!
//!     u_ldg, u_comm  (slot, var, gfpt)   gfpt fastest

comptime {
    if (@import("builtin").target.cpu.arch.isSpirV()) {
        @export(&faceCommonU, .{ .name = "face_common_u" });
    }
}

pub const PushConstants = extern struct {
    n_gfpts: u32,
    /// Interior gfpts come first; the tail is boundaries, which take their
    /// prescribed state outright rather than biasing against a side they do
    /// not have.
    n_gfpts_int: u32,
    n_vars: u32,
    /// LDG bias. `+0.5` leans fully to the left state, `-0.5` to the right.
    ldg_b: f64,
};

pub const WgSize = extern struct {
    pub const x: u32 = 128;
    pub const y: u32 = 1;
    pub const z: u32 = 1;
};

const F64Array = @SpirvType(.{ .runtime_array = f64 });
const F64Buf = extern struct { data: F64Array };

const u_ldg = @extern(*addrspace(.storage_buffer) const F64Buf, .{
    .name = "u_ldg",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 0 } },
});
const u_comm = @extern(*addrspace(.storage_buffer) F64Buf, .{
    .name = "u_comm",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 1 } },
});

const pc = @extern(*addrspace(.push_constant) const PushConstants, .{ .name = "pc" });

fn faceCommonU() callconv(.{ .spirv_kernel = .{ .x = WgSize.x, .y = WgSize.y, .z = WgSize.z } }) void {
    const gf = std.spirv.global_invocation_id[0];
    if (gf >= pc.n_gfpts) return;

    // A boundary prescribes its common state outright, which is `-0.5`: all of
    // the right (prescribed) side and none of the left.
    const b = if (gf < pc.n_gfpts_int) pc.ldg_b else -0.5;
    const wl = 0.5 + b;
    const wr = 0.5 - b;

    for (0..pc.n_vars) |n| {
        const ul = u_ldg.data[gf + pc.n_gfpts * n];
        const ur = u_ldg.data[gf + pc.n_gfpts * (n + pc.n_vars)];
        const uc = wl * ul + wr * ur;
        u_comm.data[gf + pc.n_gfpts * n] = uc;
        u_comm.data[gf + pc.n_gfpts * (n + pc.n_vars)] = uc;
    }
}

const std = @import("std");

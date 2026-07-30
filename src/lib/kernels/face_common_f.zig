//! Rusanov common normal flux at the global flux points.
//!
//! The GPU half of `Faces.rusanov`, for the inviscid Euler equations in 2D. One
//! thread per global flux point: it reads both sides, forms the upwind flux
//! between them, and writes the two sides' contributions plus the wave speed.
//!
//! The two slots get opposite signs because each element's normal points out of
//! it, and each is scaled by its own face measure `d_a`.
//!
//! Layouts:
//!
//!     u, f_comm  (slot, var, gfpt)   gfpt fastest
//!     norm       (dim, gfpt)
//!     d_a        (slot, gfpt)
//!     wave_sp    (gfpt)

comptime {
    if (@import("builtin").target.cpu.arch.isSpirV()) {
        @export(&faceCommonF, .{ .name = "face_common_f" });
    }
}

pub const PushConstants = extern struct {
    n_gfpts: u32,
    n_vars: u32,
    gamma: f64,
    /// Rusanov blending: 0 is full upwind, 1 is a plain central flux
    rus_k: f64,
};

pub const WgSize = extern struct {
    pub const x: u32 = 128;
    pub const y: u32 = 1;
    pub const z: u32 = 1;
};

const F64Array = @SpirvType(.{ .runtime_array = f64 });
const F64Buf = extern struct { data: F64Array };

const u = @extern(*addrspace(.storage_buffer) const F64Buf, .{
    .name = "u",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 0 } },
});
const norm = @extern(*addrspace(.storage_buffer) const F64Buf, .{
    .name = "norm",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 1 } },
});
const d_a = @extern(*addrspace(.storage_buffer) const F64Buf, .{
    .name = "d_a",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 2 } },
});
const f_comm = @extern(*addrspace(.storage_buffer) F64Buf, .{
    .name = "f_comm",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 3 } },
});
const wave_sp = @extern(*addrspace(.storage_buffer) F64Buf, .{
    .name = "wave_sp",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 4 } },
});

const pc = @extern(*addrspace(.push_constant) const PushConstants, .{ .name = "pc" });

const n_dims = 2;
const n_vars = 4;

/// Normal flux and the largest wave speed for one state, both of which the
/// Rusanov flux needs from each side.
fn normalFlux(s: [n_vars]f64, nrm: [n_dims]f64, gamma: f64, fn_out: *[n_vars]f64) f64 {
    const inv_rho = 1.0 / s[0];
    const mom_sq = s[1] * s[1] + s[2] * s[2];
    const press = (gamma - 1.0) * (s[3] - 0.5 * mom_sq * inv_rho);
    const enthalpy = (s[3] + press) * inv_rho;

    var vn: f64 = 0.0;
    for (0..n_dims) |d| vn += s[1 + d] * inv_rho * nrm[d];

    fn_out[0] = s[0] * vn;
    for (0..n_dims) |d| fn_out[1 + d] = s[1 + d] * vn + press * nrm[d];
    fn_out[3] = s[0] * vn * enthalpy;

    return @abs(vn) + @sqrt(gamma * press * inv_rho);
}

fn faceCommonF() callconv(.{ .spirv_kernel = .{ .x = WgSize.x, .y = WgSize.y, .z = WgSize.z } }) void {
    const gf = std.spirv.global_invocation_id[0];
    if (gf >= pc.n_gfpts) return;

    var ul: [n_vars]f64 = undefined;
    var ur: [n_vars]f64 = undefined;
    for (0..n_vars) |n| {
        ul[n] = u.data[gf + pc.n_gfpts * n];
        ur[n] = u.data[gf + pc.n_gfpts * (n + pc.n_vars)];
    }

    var nrm: [n_dims]f64 = undefined;
    for (0..n_dims) |d| nrm[d] = norm.data[gf + pc.n_gfpts * d];

    var fnl: [n_vars]f64 = undefined;
    var fnr: [n_vars]f64 = undefined;
    const eig = @max(
        normalFlux(ul, nrm, pc.gamma, &fnl),
        normalFlux(ur, nrm, pc.gamma, &fnr),
    );
    wave_sp.data[gf] = eig;

    const dissipation = 0.5 * eig * (1.0 - pc.rus_k);
    const da_l = d_a.data[gf];
    const da_r = d_a.data[gf + pc.n_gfpts];

    for (0..n_vars) |n| {
        const fc = 0.5 * (fnl[n] + fnr[n]) - dissipation * (ur[n] - ul[n]);
        f_comm.data[gf + pc.n_gfpts * n] = fc * da_l;
        f_comm.data[gf + pc.n_gfpts * (n + pc.n_vars)] = -fc * da_r;
    }
}

const std = @import("std");

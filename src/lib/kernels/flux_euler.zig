//! Inviscid Euler flux at the solution points, transformed to reference space.
//!
//! The GPU half of `Solver.fluxSpts`. One thread per `(spt, ele)` pair: each
//! reads its own conserved state and metric terms and writes its own reference
//! flux, so there is nothing to synchronize.
//!
//! Array layouts are the solver's, which put the element index last so a given
//! `(spt, var)` is contiguous over elements:
//!
//!     u_spts        (spt, var, ele)
//!     inv_jaco_spts (dim_ref, spt, dim_phys, ele)
//!     f_spts        (dim_ref, spt, var, ele)
//!
//! Viscous terms are not here. They need the solution gradient, which is a
//! second input array and a second flux to add; the CPU path still handles them.

comptime {
    if (@import("builtin").target.cpu.arch.isSpirV()) {
        @export(&fluxEuler, .{ .name = "flux_euler" });
    }
}

/// Sizes the host knows and the kernel does not. `extern` so the layout is the
/// same on both sides.
pub const PushConstants = extern struct {
    n_spts: u32,
    n_eles: u32,
    n_vars: u32,
    gamma: f64,
};

pub const WgSize = extern struct {
    pub const x: u32 = 128;
    pub const y: u32 = 1;
    pub const z: u32 = 1;
};

const VkArray = @SpirvType(.{ .runtime_array = f64 });
const VkBuf = extern struct { data: VkArray };

const u_spts = @extern(*addrspace(.storage_buffer) const VkBuf, .{
    .name = "u_spts",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 0 } },
});
const inv_jaco = @extern(*addrspace(.storage_buffer) const VkBuf, .{
    .name = "inv_jaco",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 1 } },
});
const f_spts = @extern(*addrspace(.storage_buffer) VkBuf, .{
    .name = "f_spts",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 2 } },
});

const pc = @extern(*addrspace(.push_constant) const PushConstants, .{ .name = "pc" });

/// 2D only, matching the solver. `n_vars` is 4 and `n_dims` 2; both are fixed
/// here rather than read from the push constants so the loops unroll and the
/// state stays in registers.
const n_dims = 2;
const n_vars = 4;

fn fluxEuler() callconv(.{ .spirv_kernel = .{ .x = WgSize.x, .y = WgSize.y, .z = WgSize.z } }) void {
    const tid = std.spirv.global_invocation_id[0];
    if (tid >= pc.n_spts * pc.n_eles) return;

    const spt: u32 = @divTrunc(tid, pc.n_eles);
    const ele: u32 = @mod(tid, pc.n_eles);

    // u_spts(spt, n, ele)
    const u_base = ele + pc.n_eles * (pc.n_vars * spt);
    var u: [n_vars]f64 = undefined;
    for (0..n_vars) |n| u[n] = u_spts.data[u_base + pc.n_eles * n];

    // inv_jaco(d1, spt, d2, ele): the adjugate, |J| dxi/dx
    var adj: [n_dims][n_dims]f64 = undefined;
    for (0..n_dims) |d1| {
        for (0..n_dims) |d2| {
            const idx = ele + pc.n_eles * (d2 + n_dims * (spt + pc.n_spts * d1));
            adj[d1][d2] = inv_jaco.data[idx];
        }
    }

    // Physical flux, the same expressions as `flux.convEulerNS`
    const inv_rho = 1.0 / u[0];
    const mom_sq = u[1] * u[1] + u[2] * u[2];
    const press = (pc.gamma - 1.0) * (u[3] - 0.5 * mom_sq * inv_rho);
    const enthalpy = (u[3] + press) * inv_rho;

    var f: [n_vars][n_dims]f64 = undefined;
    for (0..n_dims) |dim| {
        const vel = u[1 + dim] * inv_rho;
        f[0][dim] = u[1 + dim];
        for (0..n_dims) |d| f[1 + d][dim] = u[1 + d] * vel;
        f[3][dim] = u[1 + dim] * enthalpy;
    }
    f[1][0] += press;
    f[2][1] += press;

    // Reference-space flux: tF = adj . F, into f_spts(d1, spt, n, ele)
    for (0..n_vars) |n| {
        for (0..n_dims) |d1| {
            var sum: f64 = 0.0;
            for (0..n_dims) |d2| sum += f[n][d2] * adj[d1][d2];
            const idx = ele + pc.n_eles * (n + pc.n_vars * (spt + pc.n_spts * d1));
            f_spts.data[idx] = sum;
        }
    }
}

const std = @import("std");

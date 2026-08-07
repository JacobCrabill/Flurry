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
//! The viscous build additionally binds
//!
//!     du_spts       (dim_ref, spt, var, ele)   in: reference, out: physical
//!     jaco_det_spts (spt, ele)
//!
//! and converts the reference-space gradient to a physical one before adding the
//! Navier-Stokes viscous flux. That conversion is published back into `du_spts`,
//! which is what the face path then extrapolates.
//!
//! Built once per dimension and once per viscous/inviscid; see `n_dims` and
//! `viscous` below and the kernel loop in `build.zig`.

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
    /// Viscous builds only; ignored by the inviscid one.
    mu: f64 = 0.0,
    prandtl: f64 = 0.72,
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
const du_spts = @extern(*addrspace(.storage_buffer) VkBuf, .{
    .name = "du_spts",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 3 } },
});
const jaco_det = @extern(*addrspace(.storage_buffer) const VkBuf, .{
    .name = "jaco_det",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 4 } },
});

const pc = @extern(*addrspace(.push_constant) const PushConstants, .{ .name = "pc" });

/// The dimension this build is for, injected by `build.zig`.
///
/// Comptime rather than a push constant so every loop below unrolls and the
/// state stays in registers -- a runtime bound would leave `u`, `adj` and `f`
/// dynamically indexed, which SPIR-V puts in private memory. The host imports
/// this file for `PushConstants` and `WgSize` alone and has no `kernel_dims`
/// module, so on that side reaching for `n_dims` is a compile error rather than
/// a quietly wrong 2.
const n_dims: usize = if (@import("builtin").target.cpu.arch.isSpirV())
    @import("kernel_dims").n_dims
else
    @compileError("n_dims is device-only; the host half of this file is dimension-independent");

/// Whether this build adds the Navier-Stokes viscous flux. Comptime for the
/// same reason `n_dims` is: it decides which arrays are bound at all.
const viscous: bool = if (@import("builtin").target.cpu.arch.isSpirV())
    @import("kernel_dims").viscous
else
    @compileError("viscous is device-only");

/// Euler carries density, momentum per dimension, and total energy.
const n_vars = n_dims + 2;

/// Total energy's index in the conserved state, one past the last momentum.
const i_energy = n_dims + 1;

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
    var mom_sq: f64 = 0.0;
    for (0..n_dims) |d| mom_sq += u[1 + d] * u[1 + d];
    const ke = mom_sq * inv_rho * inv_rho;
    const press = (pc.gamma - 1.0) * (u[i_energy] - 0.5 * mom_sq * inv_rho);
    const enthalpy = (u[i_energy] + press) * inv_rho;

    var f: [n_vars][n_dims]f64 = undefined;
    for (0..n_dims) |dim| {
        const vel = u[1 + dim] * inv_rho;
        f[0][dim] = u[1 + dim];
        for (0..n_dims) |d| f[1 + d][dim] = u[1 + d] * vel;
        f[i_energy][dim] = u[1 + dim] * enthalpy;
    }
    // Pressure acts along each momentum's own direction, i.e. the diagonal
    for (0..n_dims) |d| f[1 + d][d] += press;

    if (viscous) {
        // Reference gradient -> physical, and published back for the face path.
        //
        // The whole reference gradient for a variable has to be read out before
        // any of it is written back: every physical component needs every
        // reference one. On a Cartesian mesh the adjugate is diagonal and this
        // does not show, which is exactly why it is worth being careful about.
        const inv_det = 1.0 / jaco_det.data[ele + pc.n_eles * spt];

        var du: [n_vars][n_dims]f64 = undefined;
        for (0..n_vars) |n| {
            var ref: [n_dims]f64 = undefined;
            for (0..n_dims) |d| {
                ref[d] = du_spts.data[ele + pc.n_eles * (n + pc.n_vars * (spt + pc.n_spts * d))];
            }
            for (0..n_dims) |d1| {
                var sum: f64 = 0.0;
                for (0..n_dims) |d2| sum += ref[d2] * adj[d2][d1];
                du[n][d1] = sum * inv_det;
                du_spts.data[ele + pc.n_eles * (n + pc.n_vars * (spt + pc.n_spts * d1))] = du[n][d1];
            }
        }

        viscAdd(u, du, &f, inv_rho, ke);
    }

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

/// Add the Navier-Stokes viscous flux, the same expressions as
/// `flux.viscEulerNSAdd`. `du` holds the *physical* conserved-variable
/// gradients; the primitive ones are recovered here.
///
/// Viscosity is the fixed `mu`: Sutherland's law needs a `pow`, and a case that
/// asks for it keeps the step on the CPU rather than paying for one per point.
fn viscAdd(
    u: [n_vars]f64,
    du: [n_vars][n_dims]f64,
    f: *[n_vars][n_dims]f64,
    inv_rho: f64,
    ke: f64,
) void {
    var vel: [n_dims]f64 = undefined;
    for (0..n_dims) |d| vel[d] = u[1 + d] * inv_rho;
    const e_int = u[i_energy] * inv_rho - 0.5 * ke;

    // Velocity gradients: d(rho u)/dx = rho du/dx + u drho/dx
    var dvel: [n_dims][n_dims]f64 = undefined;
    for (0..n_dims) |d| {
        for (0..n_dims) |dim| {
            dvel[d][dim] = (du[1 + d][dim] - du[0][dim] * vel[d]) * inv_rho;
        }
    }

    // Internal energy gradient, via the kinetic energy
    var de: [n_dims]f64 = undefined;
    for (0..n_dims) |dim| {
        var dke: f64 = 0.0;
        for (0..n_dims) |d| dke += vel[d] * dvel[d][dim];
        dke = 0.5 * ke * du[0][dim] + u[0] * dke;
        de[dim] = (du[i_energy][dim] - dke - du[0][dim] * e_int) * inv_rho;
    }

    // Newtonian stress with Stokes' hypothesis, so the stress is deviatoric
    var trace: f64 = 0.0;
    for (0..n_dims) |d| trace += dvel[d][d];
    const diag = trace / 3.0;

    var tau: [n_dims][n_dims]f64 = undefined;
    for (0..n_dims) |i| {
        for (0..n_dims) |j| {
            tau[i][j] = pc.mu * (dvel[i][j] + dvel[j][i]);
            if (i == j) tau[i][j] -= 2.0 * pc.mu * diag;
        }
    }

    for (0..n_dims) |dim| {
        var work: f64 = 0.0;
        for (0..n_dims) |d| {
            f.*[1 + d][dim] -= tau[d][dim];
            work += vel[d] * tau[d][dim];
        }
        f.*[i_energy][dim] -= work + (pc.mu / pc.prandtl) * pc.gamma * de[dim];
    }
}

const std = @import("std");

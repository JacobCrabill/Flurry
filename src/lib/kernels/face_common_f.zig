//! Rusanov common normal flux at the global flux points.
//!
//! The GPU half of `Faces.rusanov`, for the inviscid Euler equations. One
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
//!
//! The viscous build additionally binds
//!
//!     u_ldg  (slot, var, gfpt)        the state the viscous flux is evaluated at
//!     du     (slot, dim, var, gfpt)   the physical gradient on each side
//!
//! and accumulates the LDG common viscous flux onto `f_comm`.
//!
//! Built once per dimension and once per viscous/inviscid; see `n_dims` and
//! `viscous` below and the kernel loop in `build.zig`.

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
    // Viscous builds only; ignored by the inviscid one.
    n_gfpts_int: u32 = 0,
    mu: f64 = 0.0,
    prandtl: f64 = 0.72,
    /// LDG bias, applied here as the mirror of `face_common_u`'s
    ldg_b: f64 = 0.5,
    /// LDG penalty on the jump in the solution
    ldg_tau: f64 = 0.0,
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
const u_ldg = @extern(*addrspace(.storage_buffer) const F64Buf, .{
    .name = "u_ldg",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 5 } },
});
const du = @extern(*addrspace(.storage_buffer) const F64Buf, .{
    .name = "du",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 6 } },
});

const pc = @extern(*addrspace(.push_constant) const PushConstants, .{ .name = "pc" });

/// The dimension this build is for, injected by `build.zig`.
///
/// Comptime rather than a push constant so every loop below unrolls and the
/// state stays in registers. The host imports this file for its push-constant
/// layout alone and has no `kernel_dims` module, so on that side reaching for
/// `n_dims` is a compile error rather than a quietly wrong 2.
const n_dims: usize = if (@import("builtin").target.cpu.arch.isSpirV())
    @import("kernel_dims").n_dims
else
    @compileError("n_dims is device-only; the host half of this file is dimension-independent");

/// Whether this build adds the LDG common viscous flux.
const viscous: bool = if (@import("builtin").target.cpu.arch.isSpirV())
    @import("kernel_dims").viscous
else
    @compileError("viscous is device-only");

/// Euler carries density, momentum per dimension, and total energy.
const n_vars = n_dims + 2;

/// Total energy's index in the conserved state, one past the last momentum.
const i_energy = n_dims + 1;

/// Normal flux and the largest wave speed for one state, both of which the
/// Rusanov flux needs from each side.
fn normalFlux(s: [n_vars]f64, nrm: [n_dims]f64, gamma: f64, fn_out: *[n_vars]f64) f64 {
    const inv_rho = 1.0 / s[0];
    var mom_sq: f64 = 0.0;
    for (0..n_dims) |d| mom_sq += s[1 + d] * s[1 + d];
    const press = (gamma - 1.0) * (s[i_energy] - 0.5 * mom_sq * inv_rho);
    const enthalpy = (s[i_energy] + press) * inv_rho;

    var vn: f64 = 0.0;
    for (0..n_dims) |d| vn += s[1 + d] * inv_rho * nrm[d];

    fn_out[0] = s[0] * vn;
    for (0..n_dims) |d| fn_out[1 + d] = s[1 + d] * vn + press * nrm[d];
    fn_out[i_energy] = s[0] * vn * enthalpy;

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

    var fc: [n_vars]f64 = undefined;
    for (0..n_vars) |n| {
        fc[n] = 0.5 * (fnl[n] + fnr[n]) - dissipation * (ur[n] - ul[n]);
    }

    if (viscous) ldgAdd(gf, nrm, &fc);

    for (0..n_vars) |n| {
        f_comm.data[gf + pc.n_gfpts * n] = fc[n] * da_l;
        f_comm.data[gf + pc.n_gfpts * (n + pc.n_vars)] = -fc[n] * da_r;
    }
}

/// Add the common viscous normal flux, by LDG, the same expressions as
/// `Faces.ldgViscousAdd`.
///
/// The bias is the mirror of the one `face_common_u` applies -- where the common
/// solution leans one way, the common flux leans the other -- and a boundary
/// takes its prescribed side outright rather than biasing against a side it does
/// not have.
fn ldgAdd(gf: u32, nrm: [n_dims]f64, fc: *[n_vars]f64) void {
    // The states the viscous flux is evaluated at. `u_ldg` rather than `u`,
    // which differ at a wall that prescribes a velocity but reflects it for the
    // Riemann solve.
    var ul: [n_vars]f64 = undefined;
    var ur: [n_vars]f64 = undefined;
    for (0..n_vars) |n| {
        ul[n] = u_ldg.data[gf + pc.n_gfpts * n];
        ur[n] = u_ldg.data[gf + pc.n_gfpts * (n + pc.n_vars)];
    }

    const slot = pc.n_gfpts * pc.n_vars * n_dims;
    var dul: [n_vars][n_dims]f64 = undefined;
    var dur: [n_vars][n_dims]f64 = undefined;
    for (0..n_dims) |dim| {
        for (0..n_vars) |n| {
            const i = gf + pc.n_gfpts * (n + pc.n_vars * dim);
            dul[n][dim] = du.data[i];
            dur[n][dim] = du.data[i + slot];
        }
    }

    var fl: [n_vars][n_dims]f64 = @splat(@splat(0.0));
    var fr: [n_vars][n_dims]f64 = @splat(@splat(0.0));
    viscFlux(ul, dul, &fl);
    viscFlux(ur, dur, &fr);

    var fnl: f64 = undefined;
    var fnr: f64 = undefined;

    // A boundary has already had its gradient prescribed; that is the answer.
    const interior = gf < pc.n_gfpts_int;
    const wl: f64 = if (interior) 0.5 - pc.ldg_b else 0.0;
    const wr: f64 = if (interior) 0.5 + pc.ldg_b else 1.0;

    for (0..n_vars) |n| {
        fnl = 0.0;
        fnr = 0.0;
        for (0..n_dims) |d| {
            fnl += fl[n][d] * nrm[d];
            fnr += fr[n][d] * nrm[d];
        }
        fc[n] += wl * fnl + wr * fnr + pc.ldg_tau * (ul[n] - ur[n]);
    }
}

/// The Navier-Stokes viscous flux, the same expressions as
/// `flux.viscEulerNSAdd`. Fixed viscosity: Sutherland's law needs a `pow`, and a
/// case asking for it keeps the step on the CPU.
fn viscFlux(u_in: [n_vars]f64, dU: [n_vars][n_dims]f64, f: *[n_vars][n_dims]f64) void {
    const inv_rho = 1.0 / u_in[0];

    var vel: [n_dims]f64 = undefined;
    var ke: f64 = 0.0;
    for (0..n_dims) |d| {
        vel[d] = u_in[1 + d] * inv_rho;
        ke += vel[d] * vel[d];
    }
    const e_int = u_in[i_energy] * inv_rho - 0.5 * ke;

    var dvel: [n_dims][n_dims]f64 = undefined;
    for (0..n_dims) |d| {
        for (0..n_dims) |dim| {
            dvel[d][dim] = (dU[1 + d][dim] - dU[0][dim] * vel[d]) * inv_rho;
        }
    }

    var de: [n_dims]f64 = undefined;
    for (0..n_dims) |dim| {
        var dke: f64 = 0.0;
        for (0..n_dims) |d| dke += vel[d] * dvel[d][dim];
        dke = 0.5 * ke * dU[0][dim] + u_in[0] * dke;
        de[dim] = (dU[i_energy][dim] - dke - dU[0][dim] * e_int) * inv_rho;
    }

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

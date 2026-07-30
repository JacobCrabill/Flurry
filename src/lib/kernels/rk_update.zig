//! The explicit Runge-Kutta update.
//!
//!     dst = src - dt/|J| * sum_t coeff[t] * divF[first_stage + t]
//!
//! One thread per solution-point value. The same kernel covers all three things
//! the time loop does, which differ only in what is bound and how many terms are
//! summed:
//!
//!   * saving the solution at the start of a step -- `n_terms = 0`, so the sum
//!     is empty and this is a copy
//!   * an intermediate stage -- one term, `first_stage` selecting it
//!   * the final combination -- every stage at once, weighted by `beta`
//!
//! The Jacobian determinant appears because `divf_spts` is a reference-space
//! divergence; dividing by it returns the update to physical space.
//!
//! Layouts:
//!
//!     src, dst    (spt, var, ele)             ele fastest
//!     divf        (stage, spt, var, ele)      one block per stage
//!     jaco_det    (spt, ele)

comptime {
    if (@import("builtin").target.cpu.arch.isSpirV()) {
        @export(&rkUpdate, .{ .name = "rk_update" });
    }
}

/// The most stages any supported scheme has, and so the most terms the final
/// combination can sum.
pub const max_stages = 4;

pub const PushConstants = extern struct {
    n_spts: u32,
    n_vars: u32,
    n_eles: u32,
    /// How many stage residuals to sum; 0 makes this a copy
    n_terms: u32,
    /// The first of them
    first_stage: u32,
    dt: f64,
    coeff: [max_stages]f64,
};

pub const WgSize = extern struct {
    pub const x: u32 = 128;
    pub const y: u32 = 1;
    pub const z: u32 = 1;
};

const F64Array = @SpirvType(.{ .runtime_array = f64 });
const F64Buf = extern struct { data: F64Array };

const src = @extern(*addrspace(.storage_buffer) const F64Buf, .{
    .name = "src",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 0 } },
});
const divf = @extern(*addrspace(.storage_buffer) const F64Buf, .{
    .name = "divf",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 1 } },
});
const jaco_det = @extern(*addrspace(.storage_buffer) const F64Buf, .{
    .name = "jaco_det",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 2 } },
});
const dst = @extern(*addrspace(.storage_buffer) F64Buf, .{
    .name = "dst",
    .decoration = .{ .descriptor = .{ .set = 0, .binding = 3 } },
});

const pc = @extern(*addrspace(.push_constant) const PushConstants, .{ .name = "pc" });

fn rkUpdate() callconv(.{ .spirv_kernel = .{ .x = WgSize.x, .y = WgSize.y, .z = WgSize.z } }) void {
    const tid = std.spirv.global_invocation_id[0];
    const per_stage = pc.n_spts * pc.n_vars * pc.n_eles;
    if (tid >= per_stage) return;

    var acc: f64 = 0.0;
    if (pc.n_terms > 0) {
        // `tid` runs over (spt, var, ele) with ele fastest, so the solution
        // point index is what is left after dividing out the two inner extents.
        const ele = @mod(tid, pc.n_eles);
        const spt = @divTrunc(@divTrunc(tid, pc.n_eles), pc.n_vars);

        var sum: f64 = 0.0;
        for (0..pc.n_terms) |t| {
            sum += pc.coeff[t] * divf.data[(pc.first_stage + t) * per_stage + tid];
        }
        acc = sum * pc.dt / jaco_det.data[ele + pc.n_eles * spt];
    }

    dst.data[tid] = src.data[tid] - acc;
}

const std = @import("std");

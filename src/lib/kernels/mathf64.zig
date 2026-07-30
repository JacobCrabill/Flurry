//! Double-precision transcendentals for Vulkan compute kernels.
//!
//! Vulkan's SPIR-V has none. GLSL.std.450 `Log`, `Exp`, `Pow` and friends accept
//! 16- and 32-bit floats only, and Zig's `@log`/`@exp` lower straight to them,
//! so a kernel using them fails validation at pipeline creation:
//!
//!     GLSL.std.450 Log: expected Result Type to be a 16 or 32-bit scalar
//!     or vector float type
//!
//! Falling back to f32 is not an option in a solver whose accuracy claims are
//! the point, so these do it the long way -- exponent splitting plus a
//! polynomial, in f64 throughout.
//!
//! This file is deliberately free of anything SPIR-V-specific, so the host can
//! import it and check it against `std.math`.

const std = @import("std");

/// `x^y`, for `x > 0`.
///
/// No special cases: the callers here raise densities and pressures, which are
/// positive and finite or the solution has already diverged.
pub fn pow(x: f64, y: f64) f64 {
    if (y == 0.0) return 1.0;
    if (x == 1.0) return 1.0;
    return exp2(y * log2(x));
}

/// Base-2 logarithm of a positive, finite `x`.
pub fn log2(x: f64) f64 {
    // The exponent is peeled off by scaling rather than by reading the bits.
    // `@bitCast` between `u64` and `f64` is miscompiled by the SPIR-V backend --
    // it emits an `OpStore` whose operand type does not match the pointer, which
    // fails validation at pipeline creation -- so the bits stay out of reach.
    //
    // Linear in the binary exponent, which for the densities and pressures this
    // is used on is a handful of iterations.
    var exponent: f64 = 0.0;
    var m = x;
    while (m >= 2.0) {
        m *= 0.5;
        exponent += 1.0;
    }
    while (m < 1.0) {
        m *= 2.0;
        exponent -= 1.0;
    }

    // Centre it on 1, which brings the series argument down to |t| <= 0.1716
    // and makes it converge fast.
    if (m > std.math.sqrt2) {
        m *= 0.5;
        exponent += 1.0;
    }

    // log(m) = 2 * atanh((m-1)/(m+1)), summed to well past f64 resolution:
    // t^17/17 is under 1e-14 at the worst t.
    const t = (m - 1.0) / (m + 1.0);
    const t2 = t * t;
    var series: f64 = 1.0 / 17.0;
    inline for (.{ 15.0, 13.0, 11.0, 9.0, 7.0, 5.0, 3.0, 1.0 }) |d| {
        series = series * t2 + 1.0 / d;
    }

    const ln_m = 2.0 * t * series;
    return exponent + ln_m * std.math.log2e;
}

/// `2^y` for a finite `y` in a range the solver will actually reach.
pub fn exp2(y: f64) f64 {
    // Split into an integer part, done by building the float directly, and a
    // fraction in [-0.5, 0.5], done by series.
    const n = @round(y);
    const f = y - n;

    const z = f * std.math.ln2;

    // exp(z) by Horner over 1/k!, to k = 13: |z| <= 0.347, so the first dropped
    // term is under 1e-18.
    var series: f64 = 1.0 / 6227020800.0; // 1/13!
    inline for (.{
        479001600.0, 39916800.0, 3628800.0, 362880.0, 40320.0,
        5040.0,      720.0,      120.0,     24.0,     6.0,
        2.0,         1.0,        1.0,
    }) |d| {
        series = series * z + 1.0 / d;
    }

    // 2^n by repeated squaring, for the same reason `log2` scales rather than
    // reads bits: no `@bitCast`. At most 11 iterations over an f64's exponent
    // range.
    var scale: f64 = 1.0;
    var base: f64 = if (n < 0.0) 0.5 else 2.0;
    var k: u32 = @intFromFloat(@abs(n));
    while (k > 0) {
        if (k & 1 == 1) scale *= base;
        base *= base;
        k >>= 1;
    }

    return series * scale;
}

//! Trimmed root for the vendored `ansi_term`. Ziggy's `Ast.zig` only uses
//! `style.Style` and `format.updateStyle`, so the cursor/clear/terminal
//! modules of upstream ansi_term are not vendored.

pub const style = @import("style.zig");
pub const format = @import("format.zig");

test {
    @import("std").testing.refAllDecls(@This());
}

# Vendored ziggy

The core [ziggy](https://github.com/kristoff-it/ziggy) parser, used by
`src/lib/config.zig` to load `.cfg.ziggy` input files.

- **Upstream:** https://github.com/kristoff-it/ziggy
- **Commit:** `06d7ce8df16974e0ee7f897db50784d84f6b9f32` (`v0.2.0-1-g06d7ce8`, 2026-07-15)
- **License:** MIT, see `LICENSE`

Vendored because upstream at this commit does not build with the Zig version
this project targets (`0.17.0-dev.1454`), and its `build.zig` drags in the CLI
and LSP (`lsp_kit`, `known_folders`, `afl_kit`), none of which we use and some
of which do not compile either.

## What is here

Only the parser: `root.zig`, `Tokenizer.zig`, `Ast.zig`, `Serializer.zig`,
`Deserializer.zig`, `dynamic.zig`, `schema.zig`, `schema/`. No CLI, no LSP, no
fuzzing harness, no wasm bindings.

`src/ansi_term/` is a trimmed copy of
[ansi_term](https://github.com/kristoff-it/ansi_term) `0.3.0` — `Ast.zig` uses
it for colored rendering. Only `style.zig`, `format.zig` and `parse_style.zig`
are kept, byte-identical to upstream; `root.zig` there is ours. This leaves the
vendored parser with no external module dependencies.

> **Note:** the published `ansi_term` package excludes its license file from
> `.paths`, so no license text came with these sources and none is reproduced
> here. Confirm the terms at the upstream repository before distributing a
> build of this project.

Upstream's own test suite came along with these files and runs via
`zig build test-vendor` (also folded into `zig build test`).

## Local changes

Every edit is marked in-place with a `VENDOR FIX` or `VENDOR ADDITION`
comment. Grep for those before merging an upstream update.

### `Deserializer.zig`

1. **`@import("schema")` → `@import("schema.zig")`.** Upstream resolves this
   through a module alias declared in its own `build.zig`, so the parser could
   not be consumed as a plain module.

2. **Fixed-size array and vector deserialization.** Upstream handled `.array`
   and `.vector` in a single switch prong with a shared capture, which no
   longer compiles now that `std.builtin.Type.Array` and `Type.Vector` are
   distinct types. Because that prong never compiled, the body had accumulated
   three latent bugs, all fixed in the replacement `deserializeFixedList`:

   - it passed `d.next()` to `deserializeOne` instead of the token it had
     already read, dropping every second token;
   - an early `]` was accepted whenever `idx == len - 1`, so a list one element
     short parsed as success with a garbage last element;
   - the `.rsb` arm passed an undefined identifier `next` to `lengthMismatch`.

   Vectors additionally cannot be filled in place (they reject runtime
   indices), so values are collected into a backing array and coerced.
   Sentinel-terminated arrays are now handled too.

3. **`error.LengthMismatch` added to `Error`.** `lengthMismatch()` already
   returned it, but it was absent from the error set and from both switches in
   `Meta.reportErrors` — invisible while the only code path reaching it failed
   to compile.

4. **Tests for fixed-size arrays, vectors and sentinel arrays.** Upstream's
   `"array basics"` / `"array trailing comma"` tests deserialize into *slices*,
   so the fixed-length path had no coverage at all.

5. **`.{}` — a struct with every field defaulted — was rejected.** The `.dotlb`
   prong demanded an identifier immediately after `.{`, so an empty literal
   never reached the loop that finalizes a struct. That loop already fills in
   defaults and reports only the fields that genuinely have none, so the fix is
   to let `}` through to it. Tests added for both the accepting and the still-
   rejecting case.

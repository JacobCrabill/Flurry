const std = @import("std");
const spock_build = @import("spock");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ------ Dependencies ------

    // Spock
    const gen_vk = b.option(bool, "gen-vk", "Regenerate vulkan-zig bindings from the registry instead of using the vendored src/vulkan/vk.zig") orelse false;
    const vk_registry = b.option([]const u8, "vk-registry", "Path to the Vulkan registry (vk.xml) used by -Dgen-vk") orelse "/usr/share/vulkan/registry/vk.xml";
    const spock = b.dependency("spock", .{
        .target = target,
        .optimize = optimize,
        .@"gen-vk" = gen_vk,
        .@"vk-registry" = vk_registry,
    });

    // Ziggy: vendored (see vendor/ziggy/README.md). Only the core parser is
    // kept -- no CLI, no LSP -- so its only import is the equally-trimmed
    // ansi_term used for colored AST rendering.
    const ansi_term = b.createModule(.{
        .root_source_file = b.path("vendor/ziggy/src/ansi_term/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    const ziggy_mod = b.createModule(.{
        .root_source_file = b.path("vendor/ziggy/src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "ansi_term", .module = ansi_term },
        },
    });

    // ------ Core Library ------

    const mod = b.addModule("flurry", .{
        .root_source_file = b.path("src/lib/flurry.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "spock", .module = spock.module("spock") },
            .{ .name = "ziggy", .module = ziggy_mod },
        },
    });

    // ------ GPU kernels ------
    // Every operator application in the solver is one dense `C = A*B` over
    // row-major matrices, which is exactly what spock's prebuilt dgemm computes
    // -- so the first operators to move to the GPU need no kernel of our own.
    // Kernels written here will be added alongside via `spock.addSpirvKernel`.
    mod.addAnonymousImport("spock/dgemm.spv", .{
        .root_source_file = spock.namedLazyPath("spock/dgemm.spv"),
    });

    // Kernels of our own, compiled to SPIR-V by the same route.
    inline for (.{ "flux_euler", "face_scatter", "face_gather", "face_common_f", "face_bcs", "rk_update" }) |name| {
        const spv = spock_build.addSpirvKernel(b, .{
            .name = name,
            .root_source_file = b.path("src/lib/kernels/" ++ name ++ ".zig"),
            .optimize = optimize,
            .spock_dep = spock,
        });
        mod.addAnonymousImport(name ++ ".spv", .{ .root_source_file = spv });
    }

    // ------ Executable ------

    const exe = b.addExecutable(.{
        .name = "flurry",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "flurry", .module = mod },
            },
        }),
    });
    b.installArtifact(exe);

    // ------ Convergence study ------
    // Its own executable rather than a test: it wants ReleaseFast and takes
    // long enough that it has no business in `zig build test`.

    const convergence = b.addExecutable(.{
        .name = "flurry-convergence",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/convergence.zig"),
            .target = target,
            .optimize = if (optimize == .debug) .fast else optimize,
            .imports = &.{
                .{ .name = "flurry", .module = mod },
            },
        }),
    });
    b.installArtifact(convergence);
    const conv_cmd = b.addRunArtifact(convergence);
    conv_cmd.addPassthruArgs();
    b.step("convergence", "Measure the scheme's order of accuracy").dependOn(&conv_cmd.step);

    // ------ Run ------

    const run_step = b.step("run", "Run the app");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();

    // ------ Tests ------

    const mod_tests = b.addTest(.{ .root_module = mod });
    const run_mod_tests = b.addRunArtifact(mod_tests);
    // Config/mesh tests open fixtures by paths relative to the project root.
    run_mod_tests.setCwd(b.path("."));
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);

    // Upstream ziggy's own test suite, run against our vendored + patched
    // copy so the fixes in vendor/ziggy stay honest.
    const ziggy_tests = b.addTest(.{ .root_module = ziggy_mod });
    const run_ziggy_tests = b.addRunArtifact(ziggy_tests);
    b.step("test-vendor", "Run the vendored ziggy test suite")
        .dependOn(&run_ziggy_tests.step);
    test_step.dependOn(&run_ziggy_tests.step);
}

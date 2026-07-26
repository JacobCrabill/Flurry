const std = @import("std");

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

    // ------ Core Library ------

    const mod = b.addModule("zfd", .{
        .root_source_file = b.path("src/lib/zfd.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "spock", .module = spock.module("spock") },
        },
    });

    // ------ Executable ------

    const exe = b.addExecutable(.{
        .name = "zfd",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zfd", .module = mod },
            },
        }),
    });
    b.installArtifact(exe);

    // ------ Run ------

    const run_step = b.step("run", "Run the app");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();

    // ------ Tests ------

    const mod_tests = b.addTest(.{ .root_module = mod });
    const run_mod_tests = b.addRunArtifact(mod_tests);
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
}

# Flurry: A GPU-Accelerated Flux Reconstruction Solver in Zig

> [!NOTE]
> Under construction!

This is intended to largely a port of [Flurry-cpp](https://github.com/JacobCrabill/Flurry-cpp.git)
but in Zig and with cross-platform GPU acceleration using Vulkan.

## Running a case

```sh
zig build -Doptimize=ReleaseFast
./zig-out/bin/flurry samples/quadbox.cfg.ziggy
```

A case is a single `.cfg.ziggy` file. Either it names a Gmsh file in
`core.mesh_file`, or it has a `create_mesh` section and the Cartesian mesh is
generated on the spot — `samples/quadbox.cfg.ziggy` does the latter, so it runs
with nothing else on disk.

## Output

With `output.write_freq` set, a run writes `<output_prefix>/<output_prefix>_<iter>.vtu`
every that many steps — binary VTK XML, ready to open in ParaView. Each
high-order cell is drawn as a patch of `order²` linear sub-cells over its plot
points, and cells do not share points, so a discontinuous solution looks
discontinuous.

## What works so far

2D inviscid flow on quadrilateral meshes: Gmsh 2.2 and 4.1 input, Cartesian mesh
generation, periodic boundaries, characteristic/slip-wall/supersonic/symmetry
boundaries, explicit Euler, RK44 and Jameson-RK time stepping, and ParaView
output.

Not yet: any initial condition other than a uniform freestream, a CFL-derived
time step (`time.dt` has to be given), restarts, triangles, 3D elements, and the
viscous terms — which are written but unverified.

```sh
zig build test
```

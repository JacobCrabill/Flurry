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

## Verification

`test_case.test_case` selects an initial condition with a known exact solution,
so the error can be measured rather than inferred. With `output.error_freq` set,
a run reports the L2 error in `test_case.err_field` — see
`samples/vortex.cfg.ziggy`, which convects Shu's isentropic vortex once around a
periodic box.

```sh
zig build convergence          # refine under an exact solution, report the rate
zig build convergence -- 3     # just order 3
```

On the advected sine wave — exactly periodic, so the measured rate is the
discretization's and nothing else — orders 1–4 come out at **2.02, 3.00, 3.98,
5.00** against a design order of p+1.

The isentropic vortices reach p+1 exactly at t = 0, i.e. for the initial
collocation, but converge more slowly once integrated in time (~2.4 at order 2,
~4.3 at orders 3 and 4). That is not explained by the time step, the domain size
or the error quadrature, all of which were varied and ruled out; it is still
open.

## GPU

`--gpu` runs the operators that have been ported to Vulkan compute, via
[Spock](https://github.com/JacobCrabill/spock); everything else stays on the CPU,
which remains the reference the GPU path is checked against.

```sh
./zig-out/bin/flurry --gpu samples/vortex.cfg.ziggy
```

Ported so far: `extrapolateU`. Its operands are device-resident, so a dispatch
binds them where they lie — nothing is copied. Because that memory is
host-mapped, every un-ported operation still reads and writes it as an ordinary
slice, which is what lets the port advance one operator at a time.

It is still *slower* than the CPU, and for a reason worth knowing: Spock asks
for `host_visible | host_coherent` and takes the first matching memory type,
which on the test hardware is an uncached (write-combined) one. CPU reads from it
measured **25× slower** than from an ordinary allocation, even though a
`host_cached` type was available one index later. So an array is only moved into
device memory when the operator that binds it moves too.

Everything is f64. A device without `shaderFloat64` will not run this.

## What works so far

2D inviscid flow on quadrilateral meshes: Gmsh 2.2 and 4.1 input, Cartesian mesh
generation, periodic boundaries, characteristic/slip-wall/supersonic/symmetry
boundaries, explicit Euler, RK44 and Jameson-RK time stepping, ParaView output,
and analytic test cases with error measurement.

Not yet: a CFL-derived time step (`time.dt` has to be given), restarts,
triangles, 3D elements, and the viscous terms — which are written but unverified.

```sh
zig build test
```

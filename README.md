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

A whole time step runs on the device for 2D inviscid Euler: the eight residual
dispatches plus the Runge-Kutta update, recorded into one command buffer and
submitted once per stage. Between reports the CPU touches none of the solution
arrays. Operands are device-resident, so a dispatch binds them where they lie
and nothing is copied.

Anything the kernels do not cover falls back rather than failing:
advection-diffusion, the viscous terms, and the viscous wall conditions.

The arrays live in the device's own memory, with a host-visible block beside
each one that they are pushed to and from explicitly — at setup, at the initial
condition, and before the residual norms, the error measure or solution output.
Between those points the CPU touches nothing. A case with a CPU fallback in the
step (advection-diffusion, the viscous terms) keeps host-visible arrays instead,
since a stale block would be read.

Performance, 50 steps of the vortex sample:

| mesh | CPU | GPU |
|---|---|---|
| 32×32 | 0.68 s | **0.34 s** |
| 64×64 | 3.23 s | **0.69 s** |
| 128×128 | 13.07 s | **1.99 s** |

Host-visible memory was the whole story. The same dgemm, 16×65536 with K=16,
measured on both:

| | | |
|---|---|---|
| host-visible | 1.32 GB/s | 2.64 GFLOP/s |
| device-local | **15.09 GB/s** | **30.19 GFLOP/s** |

There is more to get — the dgemm is naive, with no tiling or shared memory —
but the memory was worth an order of magnitude on its own.

Getting there needed a fix in Spock. It asked for `host_visible | host_coherent`
and took the first matching memory type, which on the test hardware is an
uncached (write-combined) one; CPU reads from it measured **25× slower** than
from an ordinary allocation, and making one operator's operands resident cost the
whole step **8×**. Spock now prefers a cached type where the device has one, which
brings host reads back to parity.

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

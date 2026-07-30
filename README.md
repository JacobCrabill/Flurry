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

The whole residual runs on the device for 2D inviscid Euler: eight dispatches
recorded into one command buffer and submitted once. Operands are
device-resident, so a dispatch binds them where they lie and nothing is copied.
Because that memory is host-mapped, everything still on the CPU — the
Runge-Kutta update, the residual norms — reads and writes it as an ordinary
slice.

Anything the kernels do not cover falls back rather than failing:
advection-diffusion, the viscous terms, and the viscous wall conditions.

Performance, 50 steps of the vortex sample:

| mesh | CPU | GPU |
|---|---|---|
| 32×32 | 0.70 s | 0.73 s |
| 64×64 | 3.24 s | 4.33 s |
| 128×128 | 13.18 s | 17.15 s |

So it is at parity when small and ~1.3× *slower* when large, by a constant
factor rather than a widening one. Spock's buffers are host-visible by design —
its own docs say so — which means the GPU streams every array over PCIe instead
of reading VRAM. Going faster from here means device-local memory with staging,
and that only pays once nothing on the CPU needs to read these arrays: the
Runge-Kutta update is the remaining obstacle.

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

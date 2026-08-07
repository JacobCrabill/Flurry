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

A case is a single `.cfg.ziggy` file. Either it names a Gmsh file in `core.mesh_file`, or it has a
`create_mesh` section and the Cartesian mesh is generated on the spot —
`samples/quadbox.cfg.ziggy` does the latter, so it runs with nothing else on disk. `core.n_dims`
picks quads or hexes; `samples/vortex-3d.cfg.ziggy` is the 3D counterpart of
`samples/vortex.cfg.ziggy`.

## Output

With `output.write_freq` set, a run writes `<output_prefix>/<output_prefix>_<iter>.vtu` every that
many steps — binary VTK XML, ready to open in ParaView. Each high-order cell is drawn as a patch
of linear sub-cells over its plot points — `order²` quads in 2D, `order³` hexahedra in 3D — and
cells do not share points, so a discontinuous solution looks discontinuous.

## Verification

`test_case.test_case` selects an initial condition with a known exact solution, so the error can be
measured rather than inferred. With `output.error_freq` set, a run reports the L2 error in
`test_case.err_field` — see `samples/vortex.cfg.ziggy`, which convects Shu's isentropic vortex
once around a periodic box.

```sh
zig build convergence          # refine under an exact solution, report the rate
zig build convergence -- 3     # just order 3
```

On the advected sine wave — exactly periodic, so the measured rate is the discretization's and
nothing else — orders 1–4 come out at **2.02, 3.00, 3.98, 5.00** against a design order of p+1.
The same wave in 3D, `sin πx sin πy sin πz` on a triply-periodic box, gives **2.04, 3.00, 4.03** at
orders 1–3.

Switching diffusion on — the same wave, `advdiff_D = 0.1` — brings the viscous path into what is
measured: the corrected gradient, the common solution and the LDG interface flux. Orders 1–3 come
out at **2.04/1.98, 3.03/2.98, 3.95/3.96** in 2D and **2.13/2.07, 3.05/3.03, 3.93/3.95** in 3D, again
against p+1. Note that diffusion's explicit step limit goes as h² rather than h, so a viscous sweep
runs a few thousand steps per mesh.

The isentropic vortices reach p+1 exactly at t = 0, i.e. for the initial collocation, but converge
more slowly once integrated in time (~2.4 at order 2, ~4.3 at orders 3 and 4). That is not explained
by the time step, the domain size or the error quadrature, all of which were varied and ruled out;
it is still open.

## GPU

`--gpu` runs the operators that have been ported to Vulkan compute, via
[Spock](https://github.com/JacobCrabill/spock); everything else stays on the CPU, which remains the
reference the GPU path is checked against.

```sh
./zig-out/bin/flurry --gpu samples/vortex.cfg.ziggy
```

A whole time step runs on the device for Euler in either dimension, viscous or not: the residual
dispatches plus the Runge-Kutta update, recorded into one command buffer and submitted once per
stage. Between reports the CPU touches none of the solution arrays. Operands are device-resident, so
a dispatch binds them where they lie and nothing is copied. A viscous residual is fourteen
dispatches against an inviscid one's eight — the gradient at the solution points and its correction,
the common solution and its gather, one extrapolation per dimension, and the gradient scatter.

Anything the kernels do not cover falls back rather than failing: advection-diffusion, the viscous
wall conditions, and Sutherland's law — the viscous kernels take viscosity as a push constant, so a
case that wants it to follow the local temperature stays on the CPU rather than quietly running at
the freestream value.

The kernels with the Euler equations in them — the physical flux, the Rusanov flux and the boundary
states — need `n_dims` at compile time: a runtime bound would leave the conserved state and the
metric terms dynamically indexed, and SPIR-V puts a dynamically indexed local array in private
memory rather than registers. The two flux kernels carry a second axis for the same reason, since
viscous binds two more arrays and adds the stress tensor. Rather than write each variant by hand,
`build.zig` compiles them from one source apiece — four builds of the two flux kernels, two of the
boundary states — and the dispatch picks. The rest carry no equation and no dimension and are built
once.

The arrays live in the device's own memory, with a host-visible block beside each one that they are
pushed to and from explicitly — at setup, at the initial condition, and before the residual norms,
the error measure or solution output. Between those points the CPU touches nothing. A case with a
CPU fallback in the step (advection-diffusion, a viscous wall, Sutherland's law) keeps host-visible
arrays instead, since a stale block would be read.

Performance, 50 steps of the vortex sample at order 3:

| mesh       | cells  | CPU     | GPU         |
| ---------- | ------ | ------- | ----------- |
| 32×32     | 1,024  | 0.69 s  | **0.35 s**  |
| 64×64     | 4,096  | 3.25 s  | **0.59 s**  |
| 128×128   | 16,384 | 13.37 s | **1.66 s**  |
| 8×8×8    | 512    | 2.71 s  | **0.59 s**  |
| 16×16×16 | 4,096  | 23.99 s | **4.45 s**  |
| 24×24×24 | 13,824 | 83.96 s | **15.18 s** |


3D settles at around 5.5×, against 8× for the largest 2D mesh. At equal cell count (64×64 against
16×16×16) a 3D step costs about 7.5× a 2D one on either processor, which is roughly what the
shapes predict: an order-3 hex carries 64 solution points to a quad's 16 and five conserved
variables to four, and the flux kernel reads nine metric terms per point instead of four.

Switching the viscous terms on, same 50 steps at order 3:

| mesh      | cells | CPU     | GPU        |
| --------- | ----- | ------- | ---------- |
| 16×16    | 256   | 0.31 s  | **0.08 s** |
| 16×16×4 | 1,024 | 16.02 s | **3.33 s** |

The speedup holds at about 5×, so the gradient half parallelizes as well as the
rest. Per cell it costs roughly twice an inviscid step in 2D and three times in
3D, which is what the extra six dispatches and the gradient arrays buy.

Two things got it there, both measured on the same dgemm — 16×65536 with K=16, the shape the
solver dispatches:

|                                  |               |                  |
| -------------------------------- | ------------- | ---------------- |
| host-visible, thread per element | 1.32 GB/s     | 2.64 GFLOP/s     |
| device-local, thread per element | 15.09 GB/s    | 30.19 GFLOP/s    |
| device-local, tiled over rows    | **25.5 GB/s** | **51.0 GFLOP/s** |

The tiling is upstream in Spock. It is now compute-bound at ~64% of this card's double-precision
peak, so the next gain would have to come from the arithmetic rather than the memory.

Getting there needed a fix in Spock. It asked for `host_visible | host_coherent` and took the first
matching memory type, which on the test hardware is an uncached (write-combined) one; CPU reads from
it measured **25× slower** than from an ordinary allocation, and making one operator's operands
resident cost the whole step **8×**. Spock now prefers a cached type where the device has one,
which brings host reads back to parity.

Everything is f64. A device without `shaderFloat64` will not run this.

## 3D

Inviscid flow on hexahedral meshes runs the same path as 2D: the solver holds whichever element the
mesh's cell type calls for, and everything above it — transforms, faces, boundary conditions, the
RK update — is written against the dimension rather than around it.

The part with no 2D counterpart is pairing flux points across a face. A quad's face is an interval,
and two cells always meet it reversed; a hex's face is a square that two cells can meet in any of
eight relative orientations. The mesh derives the right one from the two cells' corner-vertex lists
— topology, no coordinate comparison and no second pass — and every run reports the residual
geometric mismatch:

```
 pairing   flux point mismatch 1.790e-15
```

Two checks that the 3D path is the 2D one and not a parallel reimplementation: the Shu vortex
extended along z reproduces the 2D run's residuals and L2 error to every printed digit, and its
z-momentum residual stays at ~1e-15, so nothing leaks into the third direction.

## What works so far

Inviscid and viscous flow in 2D on quadrilateral meshes and in 3D on hexahedral ones: Gmsh 2.2 and
4.1 input, Cartesian mesh generation, periodic boundaries,
characteristic/slip-wall/supersonic/symmetry boundaries, no-slip walls, explicit Euler, RK44 and
Jameson-RK time stepping, ParaView output, and analytic test cases with error measurement.

The viscous terms are verified in both dimensions: a quadratic field diffuses to its exact analytic
divergence at roundoff, and the diffusing sine wave converges at p+1. They run on the GPU too,
except with a viscous wall or Sutherland's law.

Not yet: a CFL-derived time step (`time.dt` has to be given), restarts, triangles, tets and prisms,
GPU kernels for advection-diffusion and for the viscous walls, and Sutherland's law on the device.
`zig build convergence` sweeps 2D only.

```sh
zig build test
```

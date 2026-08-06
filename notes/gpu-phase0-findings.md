# GPU backend, Phase 0: measured go/no-go

Four probes on one Spartan A100 (`gpu-a100-short`, job 28736113), plus local
work on JAX CPU. Harnesses in `notes/bench/probe*.py`, job script
`notes/bench/probes_a100.slurm`. **CPU baselines with the BLAS pinned to one
thread** -- unpinned has produced 20x-400x phantom differences in this project.

**Verdict: pass A (the spectrum) is a clear go. Pass B (the fit) is not,
for a reason that is identified and is not the hardware.**

Two reference points to keep in view. The machine being replaced is one
**128-core** sapphire node, not one core. And at float64 the A100 (9.7 TFLOP/s)
and 128 Sapphire Rapids cores (~9.8 TFLOP/s) are at arithmetic **parity** --
the GPU's structural edge here is bandwidth, ~2.0 TB/s against ~0.6-0.7.

---

## Probe 1 -- cuFFT at the real window sizes: PASS by 15x

Kill condition was: prime rungs below 20x one core. Worst prime is **294x**.

float64, per-transform `|rfft2|`:

| n | factors | CPU 1 core | A100 | vs 1 core | **vs 128 cores** |
|---|---|---|---|---|---|
| 2001 | 3·23·29 | 77.59 ms | 0.391 ms | 198x | **1.55x** |
| 1951 | **prime** | 162.71 | 0.499 | 326x | **2.55x** |
| 1601 | **prime** | 120.49 | 0.277 | 435x | **3.40x** |
| 1001 | 7·11·13 | 11.46 | 0.091 | 126x | 0.98x |
| 601 | **prime** | 15.04 | 0.034 | 446x | **3.49x** |
| 251 | **prime** | 1.84 | 0.006 | 294x | **2.30x** |

float32 is 3.35x-7.01x against the node at the same sizes.

**The prime-size worry inverts.** 16 of the 40 production rungs are prime
(`WINDOWS_KM = range(10000, 249, -250)` at `DX_M = 5000`, and `subgrid` returns
`2*(nw//2)+1`), and the concern was that cuFFT's Bluestein fallback would make
those the weak case. They are the **strongest** case -- 326-446x -- because
pocketfft's Bluestein penalty on the CPU is worse than cuFFT's. The one size
that barely wins, 1001, is the genuinely smooth one.

Padding primes to smooth sizes remains unavailable regardless: it shifts `dk`
by 5%, and a 0.33% wavenumber change was measured to move `zt` by 8%
(`bouligand-findings.md` §6).

## Probe 2 -- XLA compile time: PASS

**2.49 s per shape** on the A100 (1.36 s on CPU), compiling the real batched
fit -- the real Temme kernel, the real two JVPs, inside an LM `fori_loop`.
Roughly 200 distinct shapes (40 rungs x free-parameter configurations x prior
counts) projects to **~8 minutes** of cold compilation, against a 4 h queue.
A prior estimate of 100 min - 5 h does not reproduce.

## Probe 3 -- fixed-trip-count waste, on real L8 spectra

`vmap` has no early exit: a masked `fori_loop` runs every item to the batch
maximum. Measured over 40 rungs of `spectra_wdmam_L8.zarr`, ~370 fits each:

- median **9-11** TRF iterations at every rung, remarkably flat;
- per-rung waste mostly **2.5x-6x**;
- four outlier rungs (w250 13.2x, w1000 13.0x, w5250 17.6x, w7500 15.9x),
  each driven by one or two fits taking 139-196 iterations.

Those outliers are single stragglers holding up ~370 items, which is exactly
what compaction removes. An earlier measurement on L5 gave a worse global
11.8x; the per-rung L8 numbers are the ones to use, since a batch is per rung
anyway (`nbins` differs).

## Probe 4 -- batched fit throughput: the negative result, and its cause

| rung | nbins | scipy / 1 core | A100 (fixed 48 iters) | vs 1 core | **vs 128** |
|---|---|---|---|---|---|
| w10000 | 397 | 14.96 ms | 2.09 ms | 7.2x | **0.06x** |
| w4000 | 158 | 8.22 | 0.894 | 9.2x | **0.07x** |
| w1000 | 39 | 5.59 | 0.227 | 24.7x | **0.19x** |

`|ΔCPD|` against scipy: median 0.0004-0.0009 km, inside the archive's 0.01 km
quantisation step. The tail is large (p90 up to 0.38 km) for reasons under
"caveats" below.

**Why it is slow, which is the useful part.** At w10000 that is 43.5 µs per fit
per iteration. The arithmetic is ~4.5 MFLOP (about five Bessel evaluations over
397 bins, ~2250 flops each) -> **~103 GFLOP/s, ~1% of A100 float64 peak**. The
loop-state traffic is ~24 MB per fit-iteration -> **~550 GB/s, ~27% of peak
bandwidth**. So the kernel is **memory-bound on `fori_loop` carry**: `vmap` of
a sequential loop materialises all ten carried arrays to HBM on every one of
the continued fraction's iterations, instead of holding them in registers.

That is an implementation problem with a known fix -- a fused kernel (Pallas)
keeping loop state in registers -- not a verdict on the hardware. It is also
not in the plan, and the upside is uncertain: register residency would move the
kernel toward compute-bound, but compute-bound at float64 is only parity with
the node.

---

## Two mitigations, both measured, both smaller than hoped

**Shortening the Bessel: no lever.** Cutting the total iteration count from 150
(split at x=2: 120 CF + 30 series) to 116 (split at x=4: 64 + 52) gives
**0.97x** on CPU at *identical* accuracy (5.79e-14 either way). The split trade
is close to zero-sum and the kernel is not iteration-bound there. Worth
re-testing on GPU, where the binding constraint is different.

**Compaction: 1.35x, throttled by solver quality.** Running `chunk` iterations,
dropping the items that have stopped moving, and relaunching. Survivors per
round at chunk=16: `[192, 192, 108, 53, 31, 25]` -- **the first two rounds drop
nobody**, because the probe LM needs 32-48 iterations where scipy's TRF needs a
median of 9 (max 24 at w2000). Compaction can only harvest variance the solver
actually has.

## Caveats on probe 4, recorded so the numbers are not over-read

- Its solver is a **bound-projected LM, not TRF**, and converges 3-5x slower.
  Its median `|ΔCPD|` is fine but its tail is not: p99 was still 0.18 km after
  96 iterations where scipy needed 24. The plan calls for a faithful reflective
  TRF port for a separate and stronger reason (clip-vs-reflect was measured at
  86.3 km of CPD on 1 seed in 12, at identical misfit), and that would sharpen
  every number here.
- The compaction column in the A100 run is **void** -- that job launched before
  the fix below.

**A measurement bug worth remembering.** Compaction was first written to test
convergence on `max|Jᵀr| < 1e-8`. Measured on real archived spectra, scipy's
TRF leaves *its own optimum* at a gradient infinity-norm of ~1.2e-5 (max
1.5e-4). The test therefore never fired, nothing was ever dropped, compaction
degenerated silently into the fixed-trip-count case it exists to avoid -- and
still reported plausible timings. It now tests the relative step over a chunk.

---

## What was built and kept

`pycurious/backends/` -- `jax_bessel.py` (Temme `K_v` at real order),
`jax_model.py` (forward model, residuals, exact Jacobian), `jax_backend.py`,
gated by `tests/test_jax_backend.py` (45 tests).

- `K_v` vs scipy: **6.7e-14** worst relative error over v ∈ [0.5, 6],
  x ∈ [1e-8, 690]; exact at the degenerate half-integer orders. More accurate
  than scipy above x=698, where scipy underflows to 0 and the true value is a
  normal double.
- `dK/dv` in **forward mode** vs mpmath: **1.8e-14** worst, including every
  `a1 = 0` case.
- Forward model vs numpy: 1.4e-13. Residuals: 4.3e-14.

Three properties of that kernel are load-bearing and each cost a measurement:

1. **Fixed trip counts, never an exit on convergence.** At even `beta` the
   order is half-integer, `a1 = 0.25 - u²` is exactly zero, the CF degenerates,
   and the value converges at iteration 2 while the *tangent* has not. A loop
   masked on the value's convergence returns `dK/dv` wrong by ~1e-2 with a
   **bit-identical** value, so no test on the value can catch it. beta = 2 and
   4 are ordinary fitted values.
2. **The fixed count has a ceiling as well as a floor: [91, 174].** It needs 91
   iterations at the worst point (u = -0.15, x -> 2+) and the `c` accumulator,
   which grows like `(i-1)!`, overflows float64 at 175. `_N_CF = 120` sits
   between. In float32 the same accumulator overflows at i ≈ 35, so this
   formulation **cannot run in float32** without a rescaled recurrence.
3. **A non-positive argument must return non-finite.** `residuals` turns that
   into the penalty keeping a CPD profile above `zt`. At the default `beta = 3`
   the order is the integer 2, so `(k dz/2)^v` stays finite and positive for
   negative `dz` -- `K_v` is the *only* term supplying the non-finite value.
   Sanitising the argument for clean tangents without restoring it hands the
   optimiser a feasible region the numpy path does not have.

## Hazard: jax and `parallelise_routine` must not share a process

Importing jax makes the process multithreaded. `parallelise_routine` forks, and
CPython 3.12+ now warns that forking a multithreaded process may deadlock. The
downstream consumer forks 128 workers with `mp.get_context("fork")`. Use the
"spawn" start method, or keep jax out of any process that will fork.

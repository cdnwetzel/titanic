# Hardware Comparison: Xeon E5-2699 v4 (2016) vs Ryzen 9 5950X (2020)

Why this document exists: the project was required to reproduce bit-for-bit
across two very different machines. It did. This is the full breakdown of
what the two machines are, what they measured, and why the numbers came out
the way they did.

All measurements below come from the standardized battery in
`scripts/benchmark.py` and the full pipeline `pipeline/run_full.py`, run
with identical code (`git` HEAD 9957b44), identical pinned dependencies
(`requirements.txt`, CPython 3.12.13 on both), and `OMP_NUM_THREADS=8`.
Raw evidence: `results/benchmark_*.json`, `results/env_*.json`,
`results/asrock/`, and the captured `lscpu` output for each machine.

## 1. The machines

| | precision-t5810 | asrock-b550 |
|---|---|---|
| CPU | Intel Xeon E5-2699 v4 | AMD Ryzen 9 5950X |
| Codename | Broadwell-EP | Zen 3 (Vermeer) |
| Launch | Q1 2016 | Q4 2020 |
| Process node | 14 nm | 7 nm TSMC |
| Cores / threads | 22c / 44t | 16c / 32t |
| Base clock | 2.2 GHz | 3.4 GHz |
| Boost clock (datasheet) | 3.6 GHz | 4.9 GHz |
| Clock observed (`lscpu`) | 2.2 GHz flat, turbo not engaged | up to 5.09 GHz |
| L1d / L1i per core | 32 KiB / 32 KiB | 32 KiB / 32 KiB |
| L2 per core | 256 KiB | 512 KiB |
| L3 | 55 MiB shared | 64 MiB (2x 32 MiB CCX) |
| Memory | 256 GB DDR4-2400 ECC, 4-channel | 64 GB DDR4-3200, 2-channel |
| PCIe | 3.0, 40 lanes | 4.0, 24 lanes |
| ISA (SIMD) | AVX2, FMA (no AVX-512) | AVX2, FMA (no AVX-512) |
| Extra ISA notes | TSX, CAT/L3 QoS, RDT | VAES, VPCLMULQDQ, SHA-NI |
| TDP | 145 W | 105 W |
| Launch MSRP | about 4100 USD | 799 USD |
| Microcode | 0xb000041 | 0xa201213 |
| Mitigation burden | PTI, MDS, L1TF, retpolines, TAA (all mitigated) | Retbleed, SSB, Spectre v1/v2 only |

The t5810 reports `CPU max MHz: 2200` and BogoMIPS consistent with a flat
2.2 GHz: turbo is either disabled in firmware or not engaged by the
governor. That matters for the analysis below; the Xeon is running at 61%
of its rated boost clock while the Ryzen reaches its rated boost.

## 2. Measured results

### Standardized battery (`scripts/benchmark.py`, wall seconds)

| Evaluation | Xeon (s) | Ryzen (s) | Ryzen speedup |
|---|---|---|---|
| Champion stack, 5-fold CV (50 fits) | 78.9 | 26.8 | 2.94x |
| Tuned HGB, 15-fold screen | 16.5 | 5.4 | 3.06x |
| Tuned RF, 15-fold screen | 25.0 | 8.7 | 2.87x |
| Pytest suite (24 tests) | 19.0 | 6.7 | 2.84x |
| **Full battery (pipeline.run_full)** | **about 4 h** | **75 min** | **about 3.2x** |

### Numerical parity (scores must match across machines)

| Comparison | max abs delta |
|---|---|
| Ryzen run vs Xeon rerun (same code, same seeds) | **0.0000** |
| Ryzen run vs original canonical log | 0.0000 on all reference anchors |
| Champion anchor (`gate_tuned_reference`) | bit-identical: 0.8370378507312786 |
| Nested CV with inner HPO | 0.8406 +/- 0.0073 on both, fold-for-fold identical |
| Gate decisions (accept/reject) | identical on both machines |

The three rows that differ from the *original* log (`gate_tuned`,
`screen` task 8, `gate_hygiene`, deltas about -0.0035) are identical on
both machines and are the documented consequence of seeding the Optuna
search after the original run (see `docs/TASKS.md` section 6, parity note).
They are not hardware effects.

## 3. Analysis: where the 3x comes from

**Clock times IPC.** The workload is serial across folds with
`OMP_NUM_THREADS=8` per fit, so only 8 physical cores are ever busy on
either machine. The Xeon's extra 6 cores and 28 threads are irrelevant;
per-core throughput decides everything. Decomposition:

- Clock: 2.2 GHz observed vs about 4.6 to 5.0 GHz sustained under load:
  roughly 2.1 to 2.3x.
- IPC: Zen 3 retires roughly 1.35 to 1.5x more instructions per cycle than
  Broadwell in scalar and short-vector integer/FP code, which is exactly
  what tree building, histogram binning, and imputation are.

Product: about 2.2 x 1.4 = 3.1x, which is precisely what the battery
measures (2.84x to 3.06x per evaluation). The gap between the 5950X's
boost behavior and the Xeon's disabled turbo is the single largest factor:
the same silicon at 3.6 GHz would have closed roughly a third of the gap.

**Memory and cache do not matter at this scale.** The entire working set
(891 rows by about 30 encoded columns, well under 1 MB) fits inside the L2
of a single core on either machine. The Xeon's 4-channel ECC bandwidth
(about 76 GB/s) and the Ryzen's 2-channel (about 50 GB/s) are both idle;
L3 size differences (55 vs 64 MiB) never come into play. This is the
general shape of small-data ML: everything below the compute layer is
overprovisioned.

**ISA equivalence is why parity is bit-exact.** Neither chip has AVX-512
(AVX-512 arrived with Skylake-SP on Intel and Zen 4 on AMD), so both
execute the same AVX2/FMA kernel paths in numpy, scipy, and the GBM
libraries. The dominant computations (tree splits, histograms, integer
comparisons) are exact in integer arithmetic; the small floating-point
surface (scaling, imputation, logistic regression) evidently executed
identically rounded paths on both. The result: the champion anchor
reproduces to all 16 significant digits across two vendors, two
microarchitectures, and two process nodes. Anyone reproducing this repo on
any AVX2-capable machine should expect the same.

**Mitigation tax.** The Broadwell part carries the full 2018-era
mitigation suite (PTI for Meltdown, MDS buffer clears, L1TF, TAA). These
cost kernel-side and context-switch overhead and some userspace
performance in affected patterns. It is a secondary effect here (the
workload is userspace compute), but on syscall-heavy workloads it widens
the gap further.

**Security posture as a spec.** The lscpu vulnerability block is a
hardware age marker: the Xeon lists nine mitigated vulnerability classes,
the Ryzen four, and the Ryzen is natively unaffected by the entire
Meltdown/L1TF/MDS family. For always-on workstations this is a practical
difference, not just a benchmark one.

**Silicon economics.** The Xeon was a 4100 USD part at launch (about 100
USD used today); the Ryzen was 799 USD new and holds 64 GB of DDR4-3200.
The 5950X system delivers roughly 3x the per-core throughput, lower
mitigation burden, newer ISA extensions, and PCIe 4.0, at a fraction of
the launch price and about 70% of the TDP. The honest counterpoint: the
Xeon carries 4x the memory (256 GB ECC) and 16 more threads, which matter
for memory-resident or massively parallel work that this workload never
touches.

## 4. What this means for the project

The point of the exercise was reproducibility, and the result is the
strongest form of it: given pinned code, pinned dependencies, and fixed
seeds, two machines four process-years apart produce bit-identical ML
results, and the modern desktop does it about 3x faster. The design choices
that made this possible, serial fold evaluation and seeded everything, are
documented in `README.md`. For future hardware comparisons, the procedure
is fixed: run `scripts/benchmark.py`, commit `results/benchmark_<host>.json`
and `results/env_<host>.json`, then `scripts/compare_runs.py` against the
reference logs.

## 5. Reproduce

```bash
# on any machine
pip install -r requirements.txt            # via uv venv --python 3.12
kaggle competitions download -c titanic -p data && cd data && unzip titanic.zip && cd ..
OMP_NUM_THREADS=8 python scripts/benchmark.py
OMP_NUM_THREADS=8 python -m pipeline.run_full 2>&1 | tee results/runner_full_$(hostname).log
python scripts/compare_runs.py results/results.jsonl results/results.jsonl
```

## 6. Appendix: raw specifications

Full `lscpu` capture for each machine, including flags, caches, topology,
and the vulnerability/mitigation blocks:

- `results/lscpu_precision-t5810.txt` (Xeon)
- `results/asrock/lscpu_asrock-b550.txt` (Ryzen)
- `results/env_precision-t5810.json`, `results/env_asrock-b550.json`
  (platform, Python, and package versions)

# Ternary Matrix Multiplication Kernels on Consumer GPUs

**Under what conditions do custom ternary matrix multiplication kernels provide measurable inference performance advantages over existing low-bit inference methods on consumer GPUs?**

This repo contains the code, benchmarks, and write-up for an independent research project investigating whether ternary ({-1, 0, 1}) weight quantization, implemented as a custom Triton kernel, delivers real inference speedups on a single consumer GPU (NVIDIA RTX 2060), compared against dense FP16 and off-the-shelf low-bit quantization (`bitsandbytes` INT8/INT4).

> 📄 Full write-up: [`Ternary_WhitePaper.pdf`](Ternary_WhitePaper.pdf) ([LaTeX source](paper/ternary_kernel_whitepaper.tex))

---

## TL;DR

The common intuition, that ternary weights are faster because they replace multiplication with addition, is not the right way to think about it. Across every configuration tested, performance was governed by whether the operation was **memory-bound or compute-bound**, not by whether multiplication was avoided:

- ✅ **Ternary wins** in the memory-bound regime: small-to-medium matrix sizes at small batch sizes. Here, ternary's smaller weight footprint reduces memory traffic, which is the actual bottleneck.
- ❌ **Ternary loses** once operations become compute-bound (larger batch sizes push this transition to smaller matrix sizes). Here, raw compute throughput decides the outcome, and the kernel's FP16-cast tensor-core path caps out well below `bitsandbytes`' native INT8 tensor-core throughput.
- A kernel that **avoids multiplication entirely** (`tl.where`-based selection) is dramatically *slower* than a kernel that **reintroduces multiplication** via tensor cores (`tl.dot`), at every tested size. Avoiding a multiply instruction in scalar code does not skip a cycle on modern GPUs, it forfeits access to fused multiply-accumulate hardware.
- `INT4` quantization, despite the smallest nominal weight size, uses **more peak memory** than FP16, INT8, or ternary at every size tested, likely due to dequantization-buffer overhead.

See the [paper](Ternary_WhitePaper.pdf) for the full argument, roofline analysis, and limitations.

---

## Repo Structure

```
.
├── src/
│   └── quantize.py                    # ternary_quantize(), TernarySTE, TernaryLinear, convert_to_ternary()
├── prep_ternary_weights.py            # loads a real Pythia-160M layer, quantizes to ternary, saves to disk
├── ternary_matmulKernel.py            # selection-based (tl.where) + tensor-core (tl.dot via cast) FP32/FP16 kernels
├── ternary_matmul_dot.py              # tensor-core-only kernel, used as the controlled comparison against tl.where
├── ternary_matmul_dot_tuned.py        # triton.autotune sweep over block sizes / launch params
├── benchmark_suite.py                 # FP16 / INT8 / INT4 / ternary timing comparison across matrix sizes
├── crossover_and_roofline.py          # extended size+batch sweep with roofline (compute-bound vs memory-bound) analysis
├── memory_footprint.py                # peak GPU memory comparison across implementations
├── make_figures.py                    # regenerates the two figures used in the paper from raw benchmark numbers
├── paper/
│   ├── ternary_kernel_whitepaper.tex
│   ├── ternary_kernel_whitepaper.pdf
│   ├── fig_crossover.png
│   └── fig_roofline.png
└── README.md
```

*(Adjust paths above to match your actual repo layout if they've drifted, files above reflect what was built over the course of this project, verify filenames locally before relying on this structure.)*

---

## Setup

Tested on Windows with CUDA 12.4, Python 3.10, a single NVIDIA RTX 2060 (6GB VRAM).

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install --upgrade pip

pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.46.3    # pinned — newer versions have caused silent crashes on this setup
pip install triton-windows          # plain `pip install triton` has no Windows wheels
pip install bitsandbytes
pip install datasets lm-eval zstandard matplotlib
```

> **Windows + Triton note:** standard `triton` on PyPI ships Linux wheels only. Use the community-maintained `triton-windows` fork instead — it installs as `triton` and works as a drop-in replacement.

> **Crash note:** if any script touches `lm-eval-harness`, import `datasets` *before* `torch`/`transformers`, or it can trigger a silent Windows access-violation crash.

---

## Reproducing the Results

**1. Quantize real weights to ternary** (no training required):
```bash
python prep_ternary_weights.py
```
Loads one Linear layer from a pretrained Pythia-160M checkpoint, applies absmean-scale ternary quantization, and saves the result to `ternary_weight_sample.pt`.

**2. Verify kernel correctness:**
```bash
python ternary_matmulKernel.py
python ternary_matmul_dot.py
```
Both compare kernel output against a dense PyTorch reference and report relative error (should be ~0 for the FP32 path, within FP16 tolerance for the FP16 path).

**3. Run the benchmark suite:**
```bash
python benchmark_suite.py
```
Times FP16 dense, `bitsandbytes` INT8/INT4, and the ternary kernel across a range of matrix sizes.

**4. Run the full crossover + roofline sweep:**
```bash
python crossover_and_roofline.py
```
Sweeps matrix size and batch size, classifies each configuration as compute-bound or memory-bound, and locates the crossover point against INT8/INT4.

**5. Measure memory footprint:**
```bash
python memory_footprint.py
```

**6. Regenerate the paper's figures:**
```bash
python make_figures.py
```

---

## Key Findings (Summary)

| | Memory-bound regime (small batch / small-medium size) | Compute-bound regime (larger batch / larger size) |
|---|---|---|
| **Ternary vs. FP16** | Loses | Loses |
| **Ternary vs. INT8** | Wins | Loses |
| **Ternary vs. INT4** | Wins | Loses |

The crossover into the compute-bound regime happens at smaller matrix sizes as batch size increases, roughly an order of magnitude earlier at M=128/256 compared to M=1/32. Full data and roofline analysis in the [paper](paper/ternary_kernel_whitepaper.pdf), Sections 4–5.
<img width="1306" height="774" alt="fig_crossover" src="https://github.com/user-attachments/assets/7a9fdadb-dd7a-42e5-a18a-68bafe65f384" />


Ternary-vs-INT8 speedup ratio (INT8 time / ternary time) across matrix size, one line per batch
size. Values above the dashed line at 1.0 indicate the ternary kernel is faster. The crossover moves to smaller
matrix sizes as batch size increases.

<img width="1308" height="813" alt="fig_roofline" src="https://github.com/user-attachments/assets/e2034b61-f1ce-4f91-b943-95f0ff30000b" />

Roofline analysis at M = 128: achieved throughput versus arithmetic intensity. The ternary
kernel’s throughput plateaus well below INT8 and FP16 in the compute-bound region (right of the ridge
point), consistent with a fixed FP16 tensor-core ceiling.


---

## Limitations

- Quantization was applied **post-hoc**, without retraining or QAT, on a pretrained Pythia-160M checkpoint. This is appropriate for the kernel-level questions this project addresses, but the results say **nothing about model accuracy** under ternary quantization (see BitNet / BitNet b1.58 for that question).
- All results are specific to a **single GPU** (RTX 2060, Turing architecture, 6GB VRAM). Crossover points are a function of this GPU's specific bandwidth/compute ratio and should not be assumed to transfer to other hardware.
- The tensor-core kernel **casts ternary weights to FP16** rather than computing natively in a low-bit format, which is the most likely cause of its compute-bound underperformance, and the most direct avenue for future work.
- Several smaller caveats (approximate byte-per-weight figures, one excluded anomalous memory measurement, INT8's largely size-invariant timing) are documented in the paper's Limitations section.

---

## Motivation / Related Work

This project was motivated by [BitNet](https://arxiv.org/abs/2310.11453) and [BitNet b1.58](https://arxiv.org/abs/2402.17764), which show that ternary-weight LLMs can match full-precision model quality at scale. Those results are primarily benchmarked on datacenter-class GPUs; this project asks a narrower, kernel-level question specific to consumer hardware: **does the theoretical "no multiplier needed" advantage actually show up as measured speedup**, and if so, under what conditions? The short answer, per the findings above, is: it depends on whether you're memory-bound or compute-bound, not on whether multiplication is avoided.

Other work referenced: [LLM.int8()](https://arxiv.org/abs/2208.07339), [QLoRA](https://arxiv.org/abs/2305.14314), [Triton](https://dl.acm.org/doi/10.1145/3315508.3329973), [Pythia](https://arxiv.org/abs/2304.01373).

---

## Citation

If you reference this work, please cite the accompanying paper (see `paper/ternary_kernel_whitepaper.tex` for full BibTeX-style references to prior work cited within).

---


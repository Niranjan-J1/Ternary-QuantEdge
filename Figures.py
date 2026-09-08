import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 200,
})

# ---------------------------------------------------------------------------
# FIGURE 1: Ternary-vs-INT8 speedup ratio across matrix size, one line per M.
# Ratio = INT8_time / ternary_time. >1 means ternary is faster (wins).
# Data pulled directly from the roofline/crossover sweeps (ms).
# ---------------------------------------------------------------------------

# square sizes present in every M sweep; use K=N for a clean 1-D x-axis
sizes = [128, 768, 2048, 3584, 4096, 6144]
size_labels = ["128", "768", "2048", "3584", "4096", "6144"]

# (int8_ms, ternary_ms) per size, per M  -- from your posted tables
data = {
    1:   {"int8": [0.2303, 0.2487, 0.2769, 0.2568, 0.2803, 0.2626],
          "tern": [0.1016, 0.1062, 0.2048, 0.5080, 0.6989, 1.4516]},
    32:  {"int8": [0.2635, 0.2758, 0.2805, 0.2505, 0.2548, 0.2816],
          "tern": [0.1030, 0.1102, 0.2136, 0.5288, 0.6828, 1.1640]},
    128: {"int8": [0.2349, 0.3741, 0.2788, 0.2539, 0.3021, 0.3101],
          "tern": [0.1036, 0.1317, 0.6504, 1.5900, 1.8292, 3.9976]},
    256: {"int8": [0.2355, 0.2406, 0.2910, 0.2855, 0.3083, 0.3973],
          "tern": [0.1039, 0.2120, 1.0069, 2.7182, 3.5771, 8.0019]},
}

fig, ax = plt.subplots(figsize=(6.4, 4.0))
markers = {1: "o", 32: "s", 128: "^", 256: "D"}
for M in [1, 32, 128, 256]:
    ratio = np.array(data[M]["int8"]) / np.array(data[M]["tern"])
    ax.plot(range(len(sizes)), ratio, marker=markers[M], label=f"M = {M}", linewidth=1.8)

ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
ax.text(len(sizes) - 1, 1.03, "ternary faster above", ha="right", va="bottom", fontsize=9)
ax.text(len(sizes) - 1, 0.97, "INT8 faster below", ha="right", va="top", fontsize=9)
ax.set_xticks(range(len(sizes)))
ax.set_xticklabels(size_labels)
ax.set_xlabel("Matrix size (K = N)")
ax.set_ylabel("Speedup ratio  (INT8 time / ternary time)")
ax.set_title("Ternary kernel speedup over INT8 vs. matrix size, by batch size")
ax.set_yscale("log")
ax.legend(title="Batch size", frameon=True)
fig.tight_layout()
fig.savefig("/mnt/user-data/outputs/fig_crossover.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# FIGURE 2: Roofline. Achieved GFLOP/s vs arithmetic intensity, with the
# RTX 2060 ridge point. Uses M=128 data (spans both regimes cleanly).
# ---------------------------------------------------------------------------
PEAK_TFLOPS = 25.8 * 1e3   # GFLOP/s
PEAK_BW = 336.0            # GB/s
ridge = (PEAK_TFLOPS) / PEAK_BW  # ~76.79 FLOP/byte

# (AI, achieved GFLOP/s) at M=128 for each implementation, from your table
impl_data = {
    "FP16 dense": {"ai": [42.67, 96.0, 105.93, 113.78, 119.47, 120.47, 122.88],
                   "g": [185.9, 5086.2, 13272.4, 10383.0, 19810.5, 16142.6, 19913.0]},
    "INT8":       {"ai": [51.2, 153.6, 180.71, 204.8, 224.0, 227.56, 236.31],
                   "g": [17.9, 403.6, 2264.4, 3851.7, 12951.6, 14217.3, 31166.4]},
    "Ternary":    {"ai": [51.2, 153.6, 180.71, 204.8, 224.0, 227.56, 236.31],
                   "g": [40.5, 1146.4, 1604.7, 1650.9, 2068.2, 2348.1, 2417.4]},
}

fig, ax = plt.subplots(figsize=(6.4, 4.2))

# roofline envelope
ai_line = np.logspace(-0.5, 3, 200)
mem_roof = PEAK_BW * ai_line              # bandwidth-limited slope
roof = np.minimum(mem_roof, PEAK_TFLOPS)  # capped at peak compute
ax.plot(ai_line, roof, color="black", linewidth=1.5, label="Roofline (RTX 2060)")
ax.axvline(ridge, color="gray", linestyle=":", linewidth=1.0)
ax.text(ridge * 0.92, 6000, f"ridge point\n{ridge:.1f} FLOP/byte", fontsize=8, va="center", ha="right")

markers = {"FP16 dense": "o", "INT8": "s", "Ternary": "^"}
for name, d in impl_data.items():
    ax.scatter(d["ai"], d["g"], marker=markers[name], s=38, label=name, zorder=5)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Arithmetic intensity (FLOP / byte)")
ax.set_ylabel("Achieved throughput (GFLOP/s)")
ax.set_title("Roofline: achieved throughput vs. arithmetic intensity (M = 128)")
ax.legend(frameon=True, loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig("/mnt/user-data/outputs/fig_roofline.png", bbox_inches="tight")
plt.close(fig)

print("figures written: fig_crossover.png, fig_roofline.png")

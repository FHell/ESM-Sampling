"""Plot the outputs of a sampler run.

Usage: uv run python scripts/plot_results.py results/demo.h5 [outdir]
"""

import os
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(path, outdir):
    os.makedirs(outdir, exist_ok=True)
    with h5py.File(path, "r") as f:
        names = [n.decode() if isinstance(n, bytes) else n for n in f["quantity_names"][()]]
        Q0 = f["level0/Q"][()]           # h5py: (samples, quantities) for Julia (q, n)
        Qw = f["level0/warmup_Q"][()]
        QL = f["finest/Q"][()] if "finest/Q" in f else None
        Qa = f["alpha0/Q"][()] if "alpha0/Q" in f else None
        est = f["estimate/multilevel"][()]
        Dc = f["level0/weather_diag"][()]
        De = f["exact/weather_diag"][()]
        years = f["costmap/years"][()]
        ylab = f["costmap/year_labels"][()]
        regF = f["costmap/regimes"][()]
        acc = f["mlda/acceptance"][()] if "mlda/acceptance" in f else None

    caps = [i for i, n in enumerate(names) if n.startswith("k_")]

    # investment marginals
    fig, axes = plt.subplots(2, (len(caps) + 1) // 2, figsize=(4 * ((len(caps) + 1) // 2), 6))
    for ax, i in zip(axes.flat, caps):
        ax.hist(Q0[:, i] / 1e3, bins=40, density=True, alpha=0.6, label="level 0 (α=1)")
        if Qa is not None:
            ax.hist(Qa[:, i] / 1e3, bins=40, density=True, alpha=0.5, label="level 0 (α=0)")
        if QL is not None:
            ax.hist(QL[:, i] / 1e3, bins=20, density=True, alpha=0.5, label="finest (MLDA)")
        ax.axvline(est[i] / 1e3, color="k", lw=1, label="multilevel estimate")
        ax.set_title(names[i])
        ax.set_xlabel("GW")
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{outdir}/investments.png", dpi=130)

    # traces
    tot = names.index("total")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    ax[0].plot(np.concatenate([Qw[:, tot], Q0[:, tot]]) / 1e9)
    ax[0].axvline(len(Qw), color="k", lw=0.8)
    ax[0].set_title("total cost, level 0 (warmup | production)")
    ax[0].set_ylabel("bn €/a")
    for i in caps:
        ax[1].plot(np.concatenate([Qw[:, i], Q0[:, i]]) / 1e3, lw=0.8, label=names[i])
    ax[1].axvline(len(Qw), color="k", lw=0.8)
    ax[1].set_title("capacities, level 0")
    ax[1].set_ylabel("GW")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{outdir}/traces.png", dpi=130)

    # diagnostic §6.4: chain weather marginal vs exact weather model
    nd = Dc.shape[1]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    x = np.arange(nd)
    ax.errorbar(x - 0.15, De.mean(0), De.std(0), fmt="o", ms=3, label="exact weather model")
    ax.errorbar(x + 0.15, Dc.mean(0), Dc.std(0), fmt="o", ms=3, label="chain (α=1)")
    ax.set_title("weather marginal diagnostic: seasonal mode-1 means per field, regime frequencies")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{outdir}/weather_diagnostic.png", dpi=130)

    # weather-cost map
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    order = np.argsort(years)[::-1]
    ax[0].bar(range(len(years)), years[order] / 1e9)
    ax[0].set_xticks(range(len(years)))
    ax[0].set_xticklabels(ylab[order], rotation=90, fontsize=7)
    ax[0].set_title("F̂(x, k̄) per weather year (learned)")
    ax[0].set_ylabel("bn €/a")
    ax[1].bar(range(len(regF)), regF / 1e6)
    ax[1].set_title("cost rate per regime day")
    ax[1].set_ylabel("m€ per day")
    fig.tight_layout()
    fig.savefig(f"{outdir}/weather_cost_map.png", dpi=130)

    print("multilevel estimate:")
    for n, e in zip(names, est):
        print(f"  {n:24s} {e: .4e}")
    if acc is not None:
        print("MLDA acceptance per level:", np.round(acc, 3))
    print(f"figures in {outdir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results/figures")

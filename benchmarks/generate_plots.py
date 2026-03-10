from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_cylinder_spin():
    rows = read_csv(RESULTS / "benchmark_cylinder_spin.csv")
    time = [float(row["time"]) for row in rows]
    omega = [float(row["omega_model"]) for row in rows]
    theory = [float(row["omega_theory"]) for row in rows]
    point = [float(row["omega_point"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(time, omega, label="geometry-induced model", linewidth=2.0, color="#0b6e4f")
    ax.plot(time, theory, label="theory", linewidth=1.8, linestyle="--", color="#c84c09")
    ax.plot(time, point, label="point-contact baseline", linewidth=1.8, linestyle=":", color="#6c757d")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("angular speed about contact normal (rad/s)")
    ax.set_title("Cylinder Spin-Down From Computed Torsion Radius")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "benchmark_cylinder_spin.png", dpi=180)
    plt.close(fig)


def plot_sharp_drop():
    rows = read_csv(RESULTS / "benchmark_sharp_drop.csv")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["p_norm"]].append(row)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    colors = {"2.0000000000": "#355070", "4.0000000000": "#6d597a", "6.0000000000": "#b56576"}
    for p_norm, series in sorted(grouped.items(), key=lambda item: float(item[0])):
        time = [float(row["time"]) for row in series]
        penetration = [float(row["equivalent_penetration"]) for row in series]
        ax.plot(time, penetration, linewidth=2.0, label=rf"$p={int(round(float(p_norm)))}$", color=colors.get(p_norm))

    ax.set_xlabel("time (s)")
    ax.set_ylabel("equivalent penetration (-g_eq)")
    ax.set_title("Sharp Corner Indentation: Shape Sensitivity Versus p")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "benchmark_sharp_drop.png", dpi=180)
    plt.close(fig)


def plot_resolution_convergence():
    rows = read_csv(RESULTS / "benchmark_resolution_convergence.csv")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["p_norm"]].append(row)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    colors = {"2.0000000000": "#264653", "6.0000000000": "#e76f51"}
    for p_norm, series in sorted(grouped.items(), key=lambda item: float(item[0])):
        resolution = [float(row["resolution"]) for row in series]
        error = [max(float(row["relative_gap_error"]), 1e-14) for row in series]
        ax.loglog(
            resolution,
            error,
            marker="o",
            linewidth=2.0,
            label=rf"$p={int(round(float(p_norm)))}$",
            color=colors.get(p_norm),
        )

    ax.set_xlabel("grid resolution")
    ax.set_ylabel("relative gap error")
    ax.set_title("Resolution Convergence for Sharp-Corner Contact")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "benchmark_resolution_convergence.png", dpi=180)
    plt.close(fig)


def plot_mass_ratio_stack():
    rows = read_csv(RESULTS / "benchmark_mass_ratio_stack.csv")
    ratio = [float(row["mass_ratio"]) for row in rows]
    iterations = [float(row["iterations"]) for row in rows]
    residual = [max(float(row["residual_inf"]), 1e-14) for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.5), sharex=True)
    axes[0].plot(ratio, iterations, marker="o", linewidth=2.0, color="#005f73")
    axes[0].set_xscale("log")
    axes[0].set_ylabel("Newton iterations")
    axes[0].set_title("Two-Contact Stack: Mass-Ratio Sweep at h=0.05 s")
    axes[0].grid(True, alpha=0.25)

    axes[1].semilogy(ratio, residual, marker="o", linewidth=2.0, color="#bb3e03")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("mass ratio")
    axes[1].set_ylabel("residual inf-norm")
    axes[1].grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    fig.savefig(RESULTS / "benchmark_mass_ratio_stack.png", dpi=180)
    plt.close(fig)


def plot_engine_multicontact():
    rows = read_csv(RESULTS / "benchmark_engine_multicontact.csv")
    time = [float(row["time"]) for row in rows]
    soccp_pen = [float(row["soccp_penetration"]) for row in rows]
    penalty_pen = [float(row["penalty_penetration"]) for row in rows]
    residual = [max(float(row["soccp_residual"]), 1e-14) for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.5), sharex=True)
    axes[0].plot(time, soccp_pen, linewidth=2.0, color="#0a9396", label="SOCCP engine")
    axes[0].plot(time, penalty_pen, linewidth=2.0, color="#ae2012", linestyle="--", label="penalty baseline")
    axes[0].set_ylabel("max penetration")
    axes[0].set_title("Multi-Contact Time-Domain Comparison")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].semilogy(time, residual, linewidth=2.0, color="#5f0f40")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("SOCCP residual")
    axes[1].grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    fig.savefig(RESULTS / "benchmark_engine_multicontact.png", dpi=180)
    plt.close(fig)


def main():
    plot_cylinder_spin()
    plot_sharp_drop()
    plot_resolution_convergence()
    plot_mass_ratio_stack()
    plot_engine_multicontact()


if __name__ == "__main__":
    main()

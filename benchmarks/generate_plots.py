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


def read_single_summary(name: str) -> dict[str, str]:
    rows = read_csv(RESULTS / name)
    if len(rows) != 1:
        raise ValueError(f"expected exactly one row in {name}, got {len(rows)}")
    return rows[0]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, q)) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


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
    normal_lcp_pen = [float(row["normal_lcp_penetration"]) for row in rows]
    penalty_pen = [float(row["penalty_penetration"]) for row in rows]
    soccp_residual = [max(float(row["soccp_residual"]), 1e-14) for row in rows]
    normal_lcp_residual = [max(float(row["normal_lcp_residual"]), 1e-14) for row in rows]
    soccp_iterations = [float(row["soccp_iterations"]) for row in rows]
    normal_lcp_iterations = [float(row["normal_lcp_iterations"]) for row in rows]

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.4), sharex=True)
    axes[0].plot(time, soccp_pen, linewidth=2.0, color="#0a9396", label="SOCCP engine")
    axes[0].plot(time, normal_lcp_pen, linewidth=1.8, color="#005f73", linestyle="--", label="full normal LCP")
    axes[0].plot(time, penalty_pen, linewidth=2.0, color="#ae2012", linestyle="--", label="penalty baseline")
    axes[0].set_ylabel("max penetration")
    axes[0].set_title("Multi-Contact Time-Domain Comparison")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].semilogy(time, soccp_residual, linewidth=2.0, color="#5f0f40", label="SOCCP residual")
    axes[1].semilogy(
        time,
        normal_lcp_residual,
        linewidth=1.8,
        color="#9c6644",
        linestyle="--",
        label="normal LCP residual",
    )
    axes[1].set_ylabel("solver residual")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(frameon=False)

    axes[2].plot(time, soccp_iterations, linewidth=2.0, color="#264653", label="SOCCP iterations")
    axes[2].plot(
        time,
        normal_lcp_iterations,
        linewidth=1.8,
        color="#bb3e03",
        linestyle="--",
        label="normal LCP iterations",
    )
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("iterations / step")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(RESULTS / "benchmark_engine_multicontact.png", dpi=180)
    plt.close(fig)


def plot_engine_tripod_landing():
    rows = read_csv(RESULTS / "benchmark_engine_tripod_landing.csv")
    time = [float(row["time"]) for row in rows]
    soccp_pen = [float(row["soccp_penetration"]) for row in rows]
    normal_lcp_pen = [float(row["normal_lcp_penetration"]) for row in rows]
    penalty_pen = [float(row["penalty_penetration"]) for row in rows]
    soccp_top_y = [float(row["soccp_top_y"]) for row in rows]
    normal_lcp_top_y = [float(row["normal_lcp_top_y"]) for row in rows]
    penalty_top_y = [float(row["penalty_top_y"]) for row in rows]
    soccp_tilt = [float(row["soccp_tilt_deg"]) for row in rows]
    normal_lcp_tilt = [float(row["normal_lcp_tilt_deg"]) for row in rows]
    penalty_tilt = [float(row["penalty_tilt_deg"]) for row in rows]
    soccp_contacts = [float(row["soccp_contacts"]) for row in rows]
    normal_lcp_contacts = [float(row["normal_lcp_contacts"]) for row in rows]
    penalty_contacts = [float(row["penalty_contacts"]) for row in rows]
    soccp_residual = [max(float(row["soccp_residual"]), 1e-14) for row in rows]
    normal_lcp_residual = [max(float(row["normal_lcp_residual"]), 1e-14) for row in rows]
    soccp_iterations = [float(row["soccp_iterations"]) for row in rows]
    normal_lcp_iterations = [float(row["normal_lcp_iterations"]) for row in rows]

    fig, axes = plt.subplots(5, 1, figsize=(7.0, 10.2), sharex=True)
    axes[0].plot(time, soccp_pen, linewidth=2.0, color="#0a9396", label="SOCCP engine")
    axes[0].plot(time, normal_lcp_pen, linewidth=1.8, color="#005f73", linestyle="--", label="full normal LCP")
    axes[0].plot(time, penalty_pen, linewidth=2.0, color="#ae2012", linestyle="--", label="penalty baseline")
    axes[0].set_ylabel("max penetration")
    axes[0].set_title("3D Tripod Landing: SOCCP / NormalLCP / Tuned Penalty")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(time, soccp_top_y, linewidth=2.0, color="#264653", label="SOCCP top y")
    axes[1].plot(time, normal_lcp_top_y, linewidth=1.8, color="#2a9d8f", linestyle="--", label="normal LCP top y")
    axes[1].plot(time, penalty_top_y, linewidth=2.0, color="#e76f51", linestyle="-.", label="penalty top y")
    axes[1].set_ylabel("top body y (m)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)

    axes[2].plot(time, soccp_tilt, linewidth=2.0, color="#005f73", label="SOCCP tilt")
    axes[2].plot(time, normal_lcp_tilt, linewidth=1.8, color="#6d597a", linestyle="--", label="normal LCP tilt")
    axes[2].plot(time, penalty_tilt, linewidth=2.0, color="#bb3e03", linestyle="-.", label="penalty tilt")
    axes[2].set_ylabel("tilt angle (deg)")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(frameon=False)

    axes[3].semilogy(time, soccp_residual, linewidth=2.0, color="#5f0f40", label="SOCCP residual")
    axes[3].semilogy(time, normal_lcp_residual, linewidth=1.8, color="#9c6644", linestyle="--", label="normal LCP residual")
    axes[3].set_ylabel("raw residual")
    axes[3].grid(True, which="both", alpha=0.25)
    axes[3].legend(frameon=False)

    axes[4].step(time, soccp_contacts, where="post", linewidth=2.0, color="#264653", label="SOCCP contacts")
    axes[4].step(time, normal_lcp_contacts, where="post", linewidth=1.8, color="#2a9d8f", linestyle="--", label="normal LCP contacts")
    axes[4].step(time, penalty_contacts, where="post", linewidth=2.0, color="#9c6644", linestyle="-.", label="penalty contacts")
    axes[4].plot(time, soccp_iterations, linewidth=1.8, color="#6d597a", linestyle="--", label="SOCCP iterations")
    axes[4].plot(time, normal_lcp_iterations, linewidth=1.6, color="#bc6c25", linestyle=":", label="normal LCP iterations")
    axes[4].set_xlabel("time (s)")
    axes[4].set_ylabel("contacts / iters")
    axes[4].set_yticks([0.0, 1.0, 2.0, 3.0, 10.0, 17.0, 50.0, 100.0, 180.0])
    axes[4].grid(True, alpha=0.25)
    axes[4].legend(frameon=False, ncol=2)

    fig.tight_layout()
    fig.savefig(RESULTS / "benchmark_engine_tripod_landing.png", dpi=180)
    plt.close(fig)


def plot_penalty_tripod_tuning():
    rows = read_csv(RESULTS / "benchmark_penalty_tripod_tuning.csv")
    stiffness = [float(row["stiffness"]) for row in rows]
    damping = [float(row["damping"]) for row in rows]
    score = [float(row["score"]) for row in rows]
    penetration = [float(row["peak_penetration"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8))
    scatter_score = axes[0].scatter(stiffness, damping, c=score, s=80, cmap="viridis")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("stiffness")
    axes[0].set_ylabel("damping")
    axes[0].set_title("Penalty Tuning Score on Tripod Scene")
    axes[0].grid(True, alpha=0.25)
    fig.colorbar(scatter_score, ax=axes[0], fraction=0.046, pad=0.04)

    scatter_pen = axes[1].scatter(stiffness, damping, c=penetration, s=80, cmap="magma")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("stiffness")
    axes[1].set_ylabel("damping")
    axes[1].set_title("Peak Penetration Under Fixed dt")
    axes[1].grid(True, alpha=0.25)
    fig.colorbar(scatter_pen, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(RESULTS / "benchmark_penalty_tripod_tuning.png", dpi=180)
    plt.close(fig)


def plot_engine_oblique_box_friction():
    rows = read_csv(RESULTS / "benchmark_engine_oblique_box_friction.csv")
    time = [float(row["time"]) for row in rows]
    pen_4d = [float(row["penetration_4d"]) for row in rows]
    pen_3d = [float(row["penetration_3d"]) for row in rows]
    pen_penalty = [float(row["penetration_penalty"]) for row in rows]
    spin_4d = [float(row["spin_4d"]) for row in rows]
    spin_3d = [float(row["spin_3d"]) for row in rows]
    spin_penalty = [float(row["spin_penalty"]) for row in rows]
    slip_4d = [float(row["slip_4d"]) for row in rows]
    slip_3d = [float(row["slip_3d"]) for row in rows]
    slip_penalty = [float(row["slip_penalty"]) for row in rows]
    tilt_4d = [float(row["tilt_4d"]) for row in rows]
    tilt_3d = [float(row["tilt_3d"]) for row in rows]
    tilt_penalty = [float(row["tilt_penalty"]) for row in rows]

    fig, axes = plt.subplots(4, 1, figsize=(7.2, 8.6), sharex=True)
    axes[0].plot(time, pen_4d, linewidth=2.0, color="#0b6e4f", label="4D friction")
    axes[0].plot(time, pen_3d, linewidth=1.8, color="#005f73", linestyle="--", label="3D torsion-free")
    axes[0].plot(time, pen_penalty, linewidth=1.8, color="#ae2012", linestyle="-.", label="tuned penalty")
    axes[0].set_ylabel("penetration")
    axes[0].set_title("Oblique Box on Plane: 4D / 3D / Penalty")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, ncol=2)

    axes[1].plot(time, spin_4d, linewidth=2.0, color="#0b6e4f", label="4D friction")
    axes[1].plot(time, spin_3d, linewidth=1.8, color="#005f73", linestyle="--", label="3D torsion-free")
    axes[1].plot(time, spin_penalty, linewidth=1.8, color="#ae2012", linestyle="-.", label="tuned penalty")
    axes[1].set_ylabel("spin about normal (rad/s)")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(time, slip_4d, linewidth=2.0, color="#0b6e4f", label="4D friction")
    axes[2].plot(time, slip_3d, linewidth=1.8, color="#005f73", linestyle="--", label="3D torsion-free")
    axes[2].plot(time, slip_penalty, linewidth=1.8, color="#ae2012", linestyle="-.", label="tuned penalty")
    axes[2].set_ylabel("horizontal slip (m)")
    axes[2].grid(True, alpha=0.25)

    axes[3].plot(time, tilt_4d, linewidth=2.0, color="#0b6e4f", label="4D friction")
    axes[3].plot(time, tilt_3d, linewidth=1.8, color="#005f73", linestyle="--", label="3D torsion-free")
    axes[3].plot(time, tilt_penalty, linewidth=1.8, color="#ae2012", linestyle="-.", label="tuned penalty")
    axes[3].set_xlabel("time (s)")
    axes[3].set_ylabel("tilt (deg)")
    axes[3].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(RESULTS / "benchmark_engine_oblique_box_friction.png", dpi=180)
    plt.close(fig)


def plot_engine_torsion_compare():
    rows = read_csv(RESULTS / "benchmark_engine_torsion_compare.csv")
    time = [float(row["time"]) for row in rows]
    omega_4d = [float(row["omega_4d"]) for row in rows]
    omega_3d = [float(row["omega_3d"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(time, omega_4d, linewidth=2.0, color="#0b6e4f", label="4D friction")
    ax.plot(time, omega_3d, linewidth=2.0, color="#9c6644", linestyle="--", label="3D baseline")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("spin rate (rad/s)")
    ax.set_title("Full-Engine Torsional Friction Comparison")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "benchmark_engine_torsion_compare.png", dpi=180)
    plt.close(fig)


def plot_torsion_friction_reference():
    rows = read_csv(RESULTS / "benchmark_torsion_friction_reference.csv")
    time = [float(row["time"]) for row in rows]
    omega_ref = [float(row["omega_ref"]) for row in rows]
    omega_4d = [float(row["omega_4d"]) for row in rows]
    omega_3d = [float(row["omega_3d"]) for row in rows]
    theta_ref = [float(row["theta_ref"]) for row in rows]
    theta_4d = [float(row["theta_4d"]) for row in rows]
    theta_3d = [float(row["theta_3d"]) for row in rows]
    lambda_n_ref = [float(row["lambda_n_ref"]) for row in rows]
    lambda_n_4d = [float(row["lambda_n_4d"]) for row in rows]
    lambda_tau_ref = [float(row["lambda_tau_ref"]) for row in rows]
    lambda_tau_4d = [float(row["lambda_tau_4d"]) for row in rows]

    fig, axes = plt.subplots(4, 1, figsize=(7.2, 8.8), sharex=True)
    axes[0].plot(time, omega_ref, linewidth=1.8, color="#6c757d", linestyle=":", label="analytic reference")
    axes[0].plot(time, omega_4d, linewidth=2.0, color="#0b6e4f", label="SOCCP 4D")
    axes[0].plot(time, omega_3d, linewidth=1.8, color="#9c6644", linestyle="--", label="3D torsion-free")
    axes[0].set_ylabel("spin rate (rad/s)")
    axes[0].set_title("Controlled Torsional-Friction Reference")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, ncol=2)

    axes[1].plot(time, theta_ref, linewidth=1.8, color="#6c757d", linestyle=":", label="analytic reference")
    axes[1].plot(time, theta_4d, linewidth=2.0, color="#0b6e4f", label="SOCCP 4D")
    axes[1].plot(time, theta_3d, linewidth=1.8, color="#9c6644", linestyle="--", label="3D torsion-free")
    axes[1].set_ylabel("spin angle (rad)")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(time, lambda_n_ref, linewidth=1.8, color="#6c757d", linestyle=":", label="reference normal load")
    axes[2].plot(time, lambda_n_4d, linewidth=2.0, color="#005f73", label="SOCCP lambda_n")
    axes[2].set_ylabel("normal force")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(frameon=False)

    axes[3].plot(time, lambda_tau_ref, linewidth=1.8, color="#6c757d", linestyle=":", label="reference torsion load")
    axes[3].plot(time, lambda_tau_4d, linewidth=2.0, color="#ae2012", label="SOCCP lambda_tau")
    axes[3].set_xlabel("time (s)")
    axes[3].set_ylabel("torsion load")
    axes[3].grid(True, alpha=0.25)
    axes[3].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(RESULTS / "benchmark_torsion_friction_reference.png", dpi=180)
    plt.close(fig)


def plot_sphere_drop_compare():
    rows = read_csv(RESULTS / "benchmark_sphere_drop_compare.csv")
    time = [float(row["time"]) for row in rows]
    ref_y = [float(row["reference_height"]) for row in rows]
    soccp_y = [float(row["soccp_height"]) for row in rows]
    normal_lcp_y = [float(row["normal_lcp_height"]) for row in rows]
    penalty_y = [float(row["penalty_height"]) for row in rows]
    ref_vy = [float(row["reference_vy"]) for row in rows]
    soccp_vy = [float(row["soccp_vy"]) for row in rows]
    normal_lcp_vy = [float(row["normal_lcp_vy"]) for row in rows]
    penalty_vy = [float(row["penalty_vy"]) for row in rows]
    soccp_pen = [float(row["soccp_penetration"]) for row in rows]
    normal_lcp_pen = [float(row["normal_lcp_penetration"]) for row in rows]
    penalty_pen = [float(row["penalty_penetration"]) for row in rows]
    soccp_angle = [float(row["soccp_orientation_angle_deg"]) for row in rows]
    normal_lcp_angle = [float(row["normal_lcp_orientation_angle_deg"]) for row in rows]
    penalty_angle = [float(row["penalty_orientation_angle_deg"]) for row in rows]
    soccp_omega = [max(float(row["soccp_angular_speed"]), 1e-14) for row in rows]
    normal_lcp_omega = [max(float(row["normal_lcp_angular_speed"]), 1e-14) for row in rows]
    penalty_omega = [max(float(row["penalty_angular_speed"]), 1e-14) for row in rows]

    fig, axes = plt.subplots(5, 1, figsize=(7.2, 10.8), sharex=True)

    axes[0].plot(time, ref_y, linewidth=1.8, color="#6c757d", linestyle=":", label="free-fall reference")
    axes[0].plot(time, soccp_y, linewidth=2.0, color="#0a9396", label="SOCCP")
    axes[0].plot(time, normal_lcp_y, linewidth=1.8, color="#005f73", linestyle="--", label="full normal LCP")
    axes[0].plot(time, penalty_y, linewidth=1.8, color="#ae2012", linestyle="-.", label="penalty")
    axes[0].set_ylabel("center height (m)")
    axes[0].set_title("Sphere Drop on a Large Box: Position, Velocity, Penetration, Rotation")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, ncol=2)

    axes[1].plot(time, ref_vy, linewidth=1.8, color="#6c757d", linestyle=":", label="free-fall reference")
    axes[1].plot(time, soccp_vy, linewidth=2.0, color="#0a9396", label="SOCCP")
    axes[1].plot(time, normal_lcp_vy, linewidth=1.8, color="#005f73", linestyle="--", label="full normal LCP")
    axes[1].plot(time, penalty_vy, linewidth=1.8, color="#ae2012", linestyle="-.", label="penalty")
    axes[1].set_ylabel("vertical velocity (m/s)")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(time, soccp_pen, linewidth=2.0, color="#0a9396", label="SOCCP")
    axes[2].plot(time, normal_lcp_pen, linewidth=1.8, color="#005f73", linestyle="--", label="full normal LCP")
    axes[2].plot(time, penalty_pen, linewidth=1.8, color="#ae2012", linestyle="-.", label="penalty")
    axes[2].set_ylabel("penetration (-g_eq)")
    axes[2].grid(True, alpha=0.25)

    axes[3].plot(time, soccp_angle, linewidth=2.0, color="#0a9396", label="SOCCP")
    axes[3].plot(time, normal_lcp_angle, linewidth=1.8, color="#005f73", linestyle="--", label="full normal LCP")
    axes[3].plot(time, penalty_angle, linewidth=1.8, color="#ae2012", linestyle="-.", label="penalty")
    axes[3].set_ylabel("orientation drift (deg)")
    axes[3].grid(True, alpha=0.25)

    axes[4].semilogy(time, soccp_omega, linewidth=2.0, color="#0a9396", label="SOCCP")
    axes[4].semilogy(time, normal_lcp_omega, linewidth=1.8, color="#005f73", linestyle="--", label="full normal LCP")
    axes[4].semilogy(time, penalty_omega, linewidth=1.8, color="#ae2012", linestyle="-.", label="penalty")
    axes[4].set_xlabel("time (s)")
    axes[4].set_ylabel("angular speed (rad/s)")
    axes[4].grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    fig.savefig(RESULTS / "benchmark_sphere_drop_compare.png", dpi=180)
    plt.close(fig)


def plot_engine_sliding_friction_reference():
    rows = read_csv(RESULTS / "benchmark_engine_sliding_friction_reference.csv")
    time = [float(row["time"]) for row in rows]
    x_ref = [float(row["x_ref"]) for row in rows]
    v_ref = [float(row["v_ref"]) for row in rows]
    x_soccp = [float(row["x_soccp"]) for row in rows]
    v_soccp = [float(row["v_soccp"]) for row in rows]
    x_pgs = [float(row["x_friction_pgs"]) for row in rows]
    v_pgs = [float(row["v_friction_pgs"]) for row in rows]
    x_poly = [float(row["x_polyhedral"]) for row in rows]
    v_poly = [float(row["v_polyhedral"]) for row in rows]
    lambda_n_soccp = [float(row["lambda_n_soccp"]) for row in rows]
    lambda_n_pgs = [float(row["lambda_n_friction_pgs"]) for row in rows]
    lambda_n_poly = [float(row["lambda_n_polyhedral"]) for row in rows]
    lambda_t_soccp = [float(row["lambda_t_soccp"]) for row in rows]
    lambda_t_pgs = [float(row["lambda_t_friction_pgs"]) for row in rows]
    lambda_t_poly = [float(row["lambda_t_polyhedral"]) for row in rows]

    fig, axes = plt.subplots(4, 1, figsize=(7.2, 8.8), sharex=True)
    axes[0].plot(time, x_ref, linewidth=1.8, color="#6c757d", linestyle=":", label="analytic reference")
    axes[0].plot(time, x_soccp, linewidth=2.0, color="#0b6e4f", label="SOCCP 3D")
    axes[0].plot(time, x_pgs, linewidth=1.8, color="#005f73", linestyle="--", label="friction PGS")
    axes[0].plot(time, x_poly, linewidth=1.8, color="#9c6644", linestyle="-.", label="polyhedral friction")
    axes[0].set_ylabel("slide distance (m)")
    axes[0].set_title("Translational Friction Sanity Check: Analytic / SOCCP / PGS / Polyhedral")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, ncol=2)

    axes[1].plot(time, v_ref, linewidth=1.8, color="#6c757d", linestyle=":", label="analytic reference")
    axes[1].plot(time, v_soccp, linewidth=2.0, color="#0b6e4f", label="SOCCP 3D")
    axes[1].plot(time, v_pgs, linewidth=1.8, color="#005f73", linestyle="--", label="friction PGS")
    axes[1].plot(time, v_poly, linewidth=1.8, color="#9c6644", linestyle="-.", label="polyhedral friction")
    axes[1].set_ylabel("speed (m/s)")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(time, lambda_n_soccp, linewidth=2.0, color="#0b6e4f", label="SOCCP lambda_n")
    axes[2].plot(time, lambda_n_pgs, linewidth=1.8, color="#005f73", linestyle="--", label="PGS lambda_n")
    axes[2].plot(time, lambda_n_poly, linewidth=1.8, color="#9c6644", linestyle="-.", label="polyhedral lambda_n")
    axes[2].set_ylabel("normal impulse")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(frameon=False, ncol=2)

    axes[3].plot(time, lambda_t_soccp, linewidth=2.0, color="#0b6e4f", label="SOCCP lambda_t")
    axes[3].plot(time, lambda_t_pgs, linewidth=1.8, color="#005f73", linestyle="--", label="PGS lambda_t")
    axes[3].plot(time, lambda_t_poly, linewidth=1.8, color="#9c6644", linestyle="-.", label="polyhedral lambda_t")
    axes[3].set_xlabel("time (s)")
    axes[3].set_ylabel("tangential impulse")
    axes[3].grid(True, alpha=0.25)
    axes[3].legend(frameon=False, ncol=2)

    fig.tight_layout()
    fig.savefig(RESULTS / "benchmark_engine_sliding_friction_reference.png", dpi=180)
    plt.close(fig)


def plot_engine_oblique_box_polyhedral_compare():
    rows = read_csv(RESULTS / "benchmark_engine_oblique_box_polyhedral_compare.csv")
    time = [float(row["time"]) for row in rows]
    pen_soccp = [float(row["penetration_soccp"]) for row in rows]
    pen_poly = [float(row["penetration_polyhedral"]) for row in rows]
    spin_soccp = [float(row["spin_soccp"]) for row in rows]
    spin_poly = [float(row["spin_polyhedral"]) for row in rows]
    slip_soccp = [float(row["slip_soccp"]) for row in rows]
    slip_poly = [float(row["slip_polyhedral"]) for row in rows]
    tilt_soccp = [float(row["tilt_soccp"]) for row in rows]
    tilt_poly = [float(row["tilt_polyhedral"]) for row in rows]

    fig, axes = plt.subplots(4, 1, figsize=(7.2, 8.6), sharex=True)
    axes[0].plot(time, pen_soccp, linewidth=2.0, color="#0b6e4f", label="SOCCP 3D")
    axes[0].plot(time, pen_poly, linewidth=1.8, color="#9c6644", linestyle="--", label="polyhedral friction")
    axes[0].set_ylabel("penetration")
    axes[0].set_title("Oblique Box 3D Friction Cross-Check: SOCCP vs Polyhedral")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(time, spin_soccp, linewidth=2.0, color="#0a9396", label="SOCCP 3D")
    axes[1].plot(time, spin_poly, linewidth=1.8, color="#005f73", linestyle="--", label="polyhedral friction")
    axes[1].set_ylabel("spin about normal (rad/s)")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(time, slip_soccp, linewidth=2.0, color="#bb3e03", label="SOCCP slip")
    axes[2].plot(time, slip_poly, linewidth=1.8, color="#bc6c25", linestyle="--", label="polyhedral slip")
    axes[2].set_ylabel("horizontal slip (m)")
    axes[2].grid(True, alpha=0.25)

    axes[3].plot(time, tilt_soccp, linewidth=2.0, color="#264653", label="SOCCP 3D")
    axes[3].plot(time, tilt_poly, linewidth=1.8, color="#6d597a", linestyle="--", label="polyhedral friction")
    axes[3].set_xlabel("time (s)")
    axes[3].set_ylabel("tilt (deg)")
    axes[3].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(RESULTS / "benchmark_engine_oblique_box_polyhedral_compare.png", dpi=180)
    plt.close(fig)


def plot_4d_coupled_friction_reference():
    rows = read_csv(RESULTS / "benchmark_4d_coupled_friction_reference.csv")
    time = [float(row["time"]) for row in rows]
    vx_soccp = [float(row["vx_soccp"]) for row in rows]
    vx_pgs = [float(row["vx_4dpgs"]) for row in rows]
    vz_soccp = [float(row["vz_soccp"]) for row in rows]
    vz_pgs = [float(row["vz_4dpgs"]) for row in rows]
    omega_soccp = [float(row["omega_soccp"]) for row in rows]
    omega_pgs = [float(row["omega_4dpgs"]) for row in rows]
    slip_soccp = [float(row["slip_soccp"]) for row in rows]
    slip_pgs = [float(row["slip_4dpgs"]) for row in rows]
    lambda_tau_soccp = [float(row["lambda_tau_soccp"]) for row in rows]
    lambda_tau_pgs = [float(row["lambda_tau_4dpgs"]) for row in rows]

    fig, axes = plt.subplots(5, 1, figsize=(7.2, 9.2), sharex=True)
    axes[0].plot(time, vx_soccp, linewidth=2.0, color="#0b6e4f", label="SOCCP 4D")
    axes[0].plot(time, vx_pgs, linewidth=1.8, color="#9c6644", linestyle="--", label="independent 4D PGS")
    axes[0].set_ylabel(r"$v_x$ (m/s)")
    axes[0].set_title("Coupled 4D Friction Cross-Check: SOCCP vs Independent 4D PGS")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(time, vz_soccp, linewidth=2.0, color="#0b6e4f", label="SOCCP 4D")
    axes[1].plot(time, vz_pgs, linewidth=1.8, color="#9c6644", linestyle="--", label="independent 4D PGS")
    axes[1].set_ylabel(r"$v_z$ (m/s)")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(time, omega_soccp, linewidth=2.0, color="#0a9396", label="SOCCP 4D")
    axes[2].plot(time, omega_pgs, linewidth=1.8, color="#005f73", linestyle="--", label="independent 4D PGS")
    axes[2].set_ylabel(r"$\omega_y$ (rad/s)")
    axes[2].grid(True, alpha=0.25)

    axes[3].plot(time, slip_soccp, linewidth=2.0, color="#bb3e03", label="SOCCP slip")
    axes[3].plot(time, slip_pgs, linewidth=1.8, color="#6d597a", linestyle="--", label="4D PGS slip")
    axes[3].set_ylabel("slip (m)")
    axes[3].grid(True, alpha=0.25)

    axes[4].plot(time, lambda_tau_soccp, linewidth=2.0, color="#264653", label=r"SOCCP $\lambda_\tau$")
    axes[4].plot(time, lambda_tau_pgs, linewidth=1.8, color="#bc6c25", linestyle="--", label=r"4D PGS $\lambda_\tau$")
    axes[4].set_xlabel("time (s)")
    axes[4].set_ylabel(r"$\lambda_\tau$")
    axes[4].grid(True, alpha=0.25)
    axes[4].legend(frameon=False, ncol=2)

    fig.tight_layout()
    fig.savefig(RESULTS / "benchmark_4d_coupled_friction_reference.png", dpi=180)
    plt.close(fig)


def plot_engine_mesh_sdf_multicontact():
    rows = read_csv(RESULTS / "benchmark_engine_mesh_sdf_multicontact.csv")
    time = [float(row["time"]) for row in rows]
    upper_y_analytic = [float(row["upper_y_analytic"]) for row in rows]
    upper_y_mesh = [float(row["upper_y_mesh"]) for row in rows]
    upper_vy_analytic = [float(row["upper_vy_analytic"]) for row in rows]
    upper_vy_mesh = [float(row["upper_vy_mesh"]) for row in rows]
    penetration_analytic = [float(row["penetration_analytic"]) for row in rows]
    penetration_mesh = [float(row["penetration_mesh"]) for row in rows]
    contacts_analytic = [float(row["contacts_analytic"]) for row in rows]
    contacts_mesh = [float(row["contacts_mesh"]) for row in rows]

    fig, axes = plt.subplots(4, 1, figsize=(7.0, 8.5), sharex=True)
    axes[0].plot(time, upper_y_analytic, linewidth=2.0, color="#0a9396", label="analytic box SDFs")
    axes[0].plot(time, upper_y_mesh, linewidth=1.8, color="#bb3e03", linestyle="--", label="mesh-derived box SDFs")
    axes[0].set_ylabel("upper-box y (m)")
    axes[0].set_title("Analytic-Box vs Mesh-Box SDFs in a Two-Contact Stack")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(time, upper_vy_analytic, linewidth=2.0, color="#0a9396", label="analytic box SDFs")
    axes[1].plot(time, upper_vy_mesh, linewidth=1.8, color="#bb3e03", linestyle="--", label="mesh-derived box SDFs")
    axes[1].set_ylabel("upper-box $v_y$ (m/s)")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(time, penetration_analytic, linewidth=2.0, color="#005f73", label="analytic box SDFs")
    axes[2].plot(time, penetration_mesh, linewidth=1.8, color="#ae2012", linestyle="--", label="mesh-derived box SDFs")
    axes[2].set_ylabel("max penetration")
    axes[2].grid(True, alpha=0.25)

    axes[3].step(time, contacts_analytic, where="post", linewidth=2.0, color="#264653", label="analytic contacts")
    axes[3].step(time, contacts_mesh, where="post", linewidth=1.8, color="#bc6c25", linestyle="--", label="mesh contacts")
    axes[3].set_xlabel("time (s)")
    axes[3].set_ylabel("active contacts")
    axes[3].grid(True, alpha=0.25)
    axes[3].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(RESULTS / "benchmark_engine_mesh_sdf_multicontact.png", dpi=180)
    plt.close(fig)


def write_solver_convergence_summary():
    multicontact_rows = read_csv(RESULTS / "benchmark_engine_multicontact.csv")
    tripod_rows = read_csv(RESULTS / "benchmark_engine_tripod_landing.csv")
    torsion_rows = read_csv(RESULTS / "benchmark_engine_torsion_compare.csv")
    coupled_rows = read_csv(RESULTS / "benchmark_4d_coupled_friction_reference.csv")
    oblique_rows = read_csv(RESULTS / "benchmark_engine_oblique_box_friction.csv")
    oblique_poly_rows = read_csv(RESULTS / "benchmark_engine_oblique_box_polyhedral_compare.csv")
    mass_ratio_rows = read_csv(RESULTS / "benchmark_mass_ratio_stack.csv")

    rows = []

    multicontact_soccp_res = [float(row["soccp_residual"]) for row in multicontact_rows]
    multicontact_lcp_res = [float(row["normal_lcp_residual"]) for row in multicontact_rows]
    multicontact_soccp_scaled = [float(row["soccp_scaled_residual"]) for row in multicontact_rows]
    multicontact_lcp_scaled = [float(row["normal_lcp_scaled_residual"]) for row in multicontact_rows]
    multicontact_soccp_comp = [float(row["soccp_complementarity"]) for row in multicontact_rows]
    multicontact_lcp_comp = [float(row["normal_lcp_complementarity"]) for row in multicontact_rows]
    multicontact_soccp_iters = [float(row["soccp_iterations"]) for row in multicontact_rows]
    multicontact_lcp_iters = [float(row["normal_lcp_iterations"]) for row in multicontact_rows]
    rows.append({
        "case": "cylinder_torsion_compare",
        "branch": "4D",
        "peak_residual": f"{max(float(row['residual_4d']) for row in torsion_rows):.6e}",
        "median_residual": f"{percentile([float(row['residual_4d']) for row in torsion_rows], 0.5):.6e}",
        "p95_residual": f"{percentile([float(row['residual_4d']) for row in torsion_rows], 0.95):.6e}",
        "peak_scaled_residual": f"{max(float(row['scaled_residual_4d']) for row in torsion_rows):.6e}",
        "peak_complementarity_violation": f"{max(float(row['complementarity_4d']) for row in torsion_rows):.6e}",
        "max_iterations": f"{max(float(row['iterations_4d']) for row in torsion_rows):.0f}",
        "cap_hit_steps": f"{sum(1 for row in torsion_rows if float(row['iterations_4d']) >= 180.0):.0f}",
    })
    rows.append({
        "case": "cylinder_torsion_compare",
        "branch": "3D",
        "peak_residual": f"{max(float(row['residual_3d']) for row in torsion_rows):.6e}",
        "median_residual": f"{percentile([float(row['residual_3d']) for row in torsion_rows], 0.5):.6e}",
        "p95_residual": f"{percentile([float(row['residual_3d']) for row in torsion_rows], 0.95):.6e}",
        "peak_scaled_residual": f"{max(float(row['scaled_residual_3d']) for row in torsion_rows):.6e}",
        "peak_complementarity_violation": f"{max(float(row['complementarity_3d']) for row in torsion_rows):.6e}",
        "max_iterations": f"{max(float(row['iterations_3d']) for row in torsion_rows):.0f}",
        "cap_hit_steps": f"{sum(1 for row in torsion_rows if float(row['iterations_3d']) >= 180.0):.0f}",
    })

    rows.append({
        "case": "coupled_4d_reference",
        "branch": "SOCCP",
        "peak_residual": f"{max(float(row['raw_residual_soccp']) for row in coupled_rows):.6e}",
        "median_residual": f"{percentile([float(row['raw_residual_soccp']) for row in coupled_rows], 0.5):.6e}",
        "p95_residual": f"{percentile([float(row['raw_residual_soccp']) for row in coupled_rows], 0.95):.6e}",
        "peak_scaled_residual": f"{max(float(row['scaled_residual_soccp']) for row in coupled_rows):.6e}",
        "peak_complementarity_violation": f"{max(float(row['complementarity_soccp']) for row in coupled_rows):.6e}",
        "max_iterations": f"{max(float(row['iterations_soccp']) for row in coupled_rows):.0f}",
        "cap_hit_steps": f"{sum(1 for row in coupled_rows if float(row['iterations_soccp']) >= 160.0):.0f}",
    })
    rows.append({
        "case": "coupled_4d_reference",
        "branch": "FourDPGS",
        "peak_residual": f"{max(float(row['raw_residual_4dpgs']) for row in coupled_rows):.6e}",
        "median_residual": f"{percentile([float(row['raw_residual_4dpgs']) for row in coupled_rows], 0.5):.6e}",
        "p95_residual": f"{percentile([float(row['raw_residual_4dpgs']) for row in coupled_rows], 0.95):.6e}",
        "peak_scaled_residual": f"{max(float(row['scaled_residual_4dpgs']) for row in coupled_rows):.6e}",
        "peak_complementarity_violation": f"{max(float(row['complementarity_4dpgs']) for row in coupled_rows):.6e}",
        "max_iterations": f"{max(float(row['iterations_4dpgs']) for row in coupled_rows):.0f}",
        "cap_hit_steps": f"{sum(1 for row in coupled_rows if float(row['iterations_4dpgs']) >= 400.0):.0f}",
    })

    rows.append({
        "case": "two_contact_normal_only",
        "branch": "SOCCP",
        "peak_residual": f"{max(multicontact_soccp_res):.6e}",
        "median_residual": f"{percentile(multicontact_soccp_res, 0.5):.6e}",
        "p95_residual": f"{percentile(multicontact_soccp_res, 0.95):.6e}",
        "peak_scaled_residual": f"{max(multicontact_soccp_scaled):.6e}",
        "peak_complementarity_violation": f"{max(multicontact_soccp_comp):.6e}",
        "max_iterations": f"{max(multicontact_soccp_iters):.0f}",
        "cap_hit_steps": f"{sum(1 for value in multicontact_soccp_iters if value >= 160.0):.0f}",
    })
    rows.append({
        "case": "two_contact_normal_only",
        "branch": "NormalLCP",
        "peak_residual": f"{max(multicontact_lcp_res):.6e}",
        "median_residual": f"{percentile(multicontact_lcp_res, 0.5):.6e}",
        "p95_residual": f"{percentile(multicontact_lcp_res, 0.95):.6e}",
        "peak_scaled_residual": f"{max(multicontact_lcp_scaled):.6e}",
        "peak_complementarity_violation": f"{max(multicontact_lcp_comp):.6e}",
        "max_iterations": f"{max(multicontact_lcp_iters):.0f}",
        "cap_hit_steps": f"{sum(1 for value in multicontact_lcp_iters if value >= 160.0):.0f}",
    })

    tripod_res = [float(row["soccp_residual"]) for row in tripod_rows]
    tripod_iters = [float(row["soccp_iterations"]) for row in tripod_rows]
    tripod_scaled = [float(row["soccp_scaled_residual"]) for row in tripod_rows]
    tripod_comp = [float(row["soccp_complementarity"]) for row in tripod_rows]
    tripod_lcp_res = [float(row["normal_lcp_residual"]) for row in tripod_rows]
    tripod_lcp_iters = [float(row["normal_lcp_iterations"]) for row in tripod_rows]
    tripod_lcp_scaled = [float(row["normal_lcp_scaled_residual"]) for row in tripod_rows]
    tripod_lcp_comp = [float(row["normal_lcp_complementarity"]) for row in tripod_rows]
    rows.append({
        "case": "tripod_normal_only",
        "branch": "SOCCP",
        "peak_residual": f"{max(tripod_res):.6e}",
        "median_residual": f"{percentile(tripod_res, 0.5):.6e}",
        "p95_residual": f"{percentile(tripod_res, 0.95):.6e}",
        "peak_scaled_residual": f"{max(tripod_scaled):.6e}",
        "peak_complementarity_violation": f"{max(tripod_comp):.6e}",
        "max_iterations": f"{max(tripod_iters):.0f}",
        "cap_hit_steps": f"{sum(1 for value in tripod_iters if value >= 180.0):.0f}",
    })
    rows.append({
        "case": "tripod_normal_only",
        "branch": "NormalLCP",
        "peak_residual": f"{max(tripod_lcp_res):.6e}",
        "median_residual": f"{percentile(tripod_lcp_res, 0.5):.6e}",
        "p95_residual": f"{percentile(tripod_lcp_res, 0.95):.6e}",
        "peak_scaled_residual": f"{max(tripod_lcp_scaled):.6e}",
        "peak_complementarity_violation": f"{max(tripod_lcp_comp):.6e}",
        "max_iterations": f"{max(tripod_lcp_iters):.0f}",
        "cap_hit_steps": f"{sum(1 for value in tripod_lcp_iters if value >= 180.0):.0f}",
    })

    oblique_res_4d = [float(row["residual_4d"]) for row in oblique_rows]
    oblique_res_3d = [float(row["residual_3d"]) for row in oblique_rows]
    oblique_scaled_4d = [float(row["scaled_residual_4d"]) for row in oblique_rows]
    oblique_scaled_3d = [float(row["scaled_residual_3d"]) for row in oblique_rows]
    oblique_comp_4d = [float(row["complementarity_4d"]) for row in oblique_rows]
    oblique_comp_3d = [float(row["complementarity_3d"]) for row in oblique_rows]
    oblique_iters_4d = [float(row["iterations_4d"]) for row in oblique_rows]
    oblique_iters_3d = [float(row["iterations_3d"]) for row in oblique_rows]
    rows.append({
        "case": "oblique_box_friction",
        "branch": "4D",
        "peak_residual": f"{max(oblique_res_4d):.6e}",
        "median_residual": f"{percentile(oblique_res_4d, 0.5):.6e}",
        "p95_residual": f"{percentile(oblique_res_4d, 0.95):.6e}",
        "peak_scaled_residual": f"{max(oblique_scaled_4d):.6e}",
        "peak_complementarity_violation": f"{max(oblique_comp_4d):.6e}",
        "max_iterations": f"{max(oblique_iters_4d):.0f}",
        "cap_hit_steps": f"{sum(1 for value in oblique_iters_4d if value >= 180.0):.0f}",
    })
    rows.append({
        "case": "oblique_box_friction",
        "branch": "3D",
        "peak_residual": f"{max(oblique_res_3d):.6e}",
        "median_residual": f"{percentile(oblique_res_3d, 0.5):.6e}",
        "p95_residual": f"{percentile(oblique_res_3d, 0.95):.6e}",
        "peak_scaled_residual": f"{max(oblique_scaled_3d):.6e}",
        "peak_complementarity_violation": f"{max(oblique_comp_3d):.6e}",
        "max_iterations": f"{max(oblique_iters_3d):.0f}",
        "cap_hit_steps": f"{sum(1 for value in oblique_iters_3d if value >= 180.0):.0f}",
    })

    oblique_poly_soccp_res = [float(row["residual_soccp"]) for row in oblique_poly_rows]
    oblique_poly_poly_res = [float(row["residual_polyhedral"]) for row in oblique_poly_rows]
    oblique_poly_soccp_scaled = [float(row["scaled_residual_soccp"]) for row in oblique_poly_rows]
    oblique_poly_poly_scaled = [float(row["scaled_residual_polyhedral"]) for row in oblique_poly_rows]
    oblique_poly_soccp_comp = [float(row["complementarity_soccp"]) for row in oblique_poly_rows]
    oblique_poly_poly_comp = [float(row["complementarity_polyhedral"]) for row in oblique_poly_rows]
    oblique_poly_soccp_iters = [float(row["iterations_soccp"]) for row in oblique_poly_rows]
    oblique_poly_poly_iters = [float(row["iterations_polyhedral"]) for row in oblique_poly_rows]
    rows.append({
        "case": "oblique_box_polyhedral",
        "branch": "SOCCP-3D",
        "peak_residual": f"{max(oblique_poly_soccp_res):.6e}",
        "median_residual": f"{percentile(oblique_poly_soccp_res, 0.5):.6e}",
        "p95_residual": f"{percentile(oblique_poly_soccp_res, 0.95):.6e}",
        "peak_scaled_residual": f"{max(oblique_poly_soccp_scaled):.6e}",
        "peak_complementarity_violation": f"{max(oblique_poly_soccp_comp):.6e}",
        "max_iterations": f"{max(oblique_poly_soccp_iters):.0f}",
        "cap_hit_steps": f"{sum(1 for value in oblique_poly_soccp_iters if value >= 180.0):.0f}",
    })
    rows.append({
        "case": "oblique_box_polyhedral",
        "branch": "Polyhedral",
        "peak_residual": f"{max(oblique_poly_poly_res):.6e}",
        "median_residual": f"{percentile(oblique_poly_poly_res, 0.5):.6e}",
        "p95_residual": f"{percentile(oblique_poly_poly_res, 0.95):.6e}",
        "peak_scaled_residual": f"{max(oblique_poly_poly_scaled):.6e}",
        "peak_complementarity_violation": f"{max(oblique_poly_poly_comp):.6e}",
        "max_iterations": f"{max(oblique_poly_poly_iters):.0f}",
        "cap_hit_steps": f"{sum(1 for value in oblique_poly_poly_iters if value >= 180.0):.0f}",
    })

    mass_ratio_res = [float(row["residual_inf"]) for row in mass_ratio_rows]
    mass_ratio_scaled = [float(row["scaled_residual"]) for row in mass_ratio_rows]
    mass_ratio_comp = [float(row["complementarity_violation"]) for row in mass_ratio_rows]
    mass_ratio_iters = [float(row["iterations"]) for row in mass_ratio_rows]
    rows.append({
        "case": "mass_ratio_sweep",
        "branch": "SOCCP",
        "peak_residual": f"{max(mass_ratio_res):.6e}",
        "median_residual": f"{percentile(mass_ratio_res, 0.5):.6e}",
        "p95_residual": f"{percentile(mass_ratio_res, 0.95):.6e}",
        "peak_scaled_residual": f"{max(mass_ratio_scaled):.6e}",
        "peak_complementarity_violation": f"{max(mass_ratio_comp):.6e}",
        "max_iterations": f"{max(mass_ratio_iters):.0f}",
        "cap_hit_steps": "0",
    })

    with (RESULTS / "solver_convergence_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case",
                "branch",
                "peak_residual",
                "median_residual",
                "p95_residual",
                "peak_scaled_residual",
                "peak_complementarity_violation",
                "max_iterations",
                "cap_hit_steps",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_reviewer_comparison_summary():
    cylinder = read_single_summary("benchmark_cylinder_spin_summary.csv")
    torsion_ref = read_single_summary("benchmark_torsion_friction_reference_summary.csv")
    coupled_ref = read_single_summary("benchmark_4d_coupled_friction_reference_summary.csv")
    torsion = read_single_summary("benchmark_engine_torsion_compare_summary.csv")
    sharp_rows = read_csv(RESULTS / "benchmark_sharp_drop_summary.csv")
    sharp_by_p = {int(round(float(row["p_norm"]))): row for row in sharp_rows}
    resolution = read_single_summary("benchmark_resolution_convergence_summary.csv")
    multicontact = read_single_summary("benchmark_engine_multicontact_summary.csv")
    mass_ratio = read_single_summary("benchmark_mass_ratio_stack_summary.csv")
    tripod = read_single_summary("benchmark_engine_tripod_landing_summary.csv")
    tripod_tuning = read_single_summary("benchmark_penalty_tripod_tuning_summary.csv")
    oblique = read_single_summary("benchmark_engine_oblique_box_friction_summary.csv")
    oblique_poly = read_single_summary("benchmark_engine_oblique_box_polyhedral_compare_summary.csv")
    mesh_sdf = read_single_summary("benchmark_engine_mesh_sdf_multicontact_summary.csv")

    p2_final = float(sharp_by_p[2]["final_equivalent_penetration"])
    p4_final = float(sharp_by_p[4]["final_equivalent_penetration"])
    p6_final = float(sharp_by_p[6]["final_equivalent_penetration"])
    p4_gain = (p4_final / p2_final - 1.0) * 100.0
    p6_gain = (p6_final / p2_final - 1.0) * 100.0

    rows = [
        {
            "case": "Case 1",
            "comparison": "torsion radius vs theory + point baseline",
            "primary_metric": (
                f"radius error={float(cylinder['relative_radius_error']) * 100.0:.2f}%; "
                f"stop-time error={float(cylinder['relative_stop_error']) * 100.0:.2f}%"
            ),
            "secondary_metric": f"point baseline final spin={float(cylinder['final_omega_point']):.2f} rad/s",
            "takeaway": "finite-contact torsion scale is necessary and quantitatively calibrated",
        },
        {
            "case": "Case 2",
            "comparison": "controlled torsional reference: analytic vs 4D vs 3D",
            "primary_metric": (
                f"stop time={float(torsion_ref['stop_time_ref']):.3f} / "
                f"{float(torsion_ref['stop_time_4d']):.3f} / "
                f"{float(torsion_ref['stop_time_3d']):.3f} s"
            ),
            "secondary_metric": (
                f"max omega error={float(torsion_ref['max_omega_error_4d']):.1e}; "
                f"active lambda_tau error={float(torsion_ref['max_lambda_tau_error_active']):.1e}"
            ),
            "takeaway": "the 4D torsional branch reproduces the controlled analytic spin-down law before moving to full-engine scenes",
        },
        {
            "case": "Case 3",
            "comparison": "coupled 4D solver cross-check: SOCCP vs independent 4D PGS",
            "primary_metric": (
                f"max slip/yaw diff={float(coupled_ref['max_slip_diff']):.3e} / "
                f"{float(coupled_ref['max_yaw_diff']):.3e}"
            ),
            "secondary_metric": (
                f"max lambda_t diff={float(coupled_ref['max_lambda_t_diff']):.3e}; "
                f"peak scaled residuals={float(coupled_ref['peak_scaled_residual_soccp']):.1e}/"
                f"{float(coupled_ref['peak_scaled_residual_4dpgs']):.1e}"
            ),
            "takeaway": "when translation, yaw and torsion are simultaneously active, the 4D SOCCP branch remains consistent with an independent 4D projected solver rather than only with its own controlled torsion limit",
        },
        {
            "case": "Case 4",
            "comparison": "full-engine 4D friction vs 3D torsion-free baseline",
            "primary_metric": (
                f"final spin={float(torsion['final_omega_4d']):.2e} vs "
                f"{float(torsion['final_omega_3d']):.2f} rad/s"
            ),
            "secondary_metric": f"residual spin reduction={float(torsion['spin_decay_ratio']) * 100.0:.3f}%",
            "takeaway": "torsional impulse changes the full-engine dynamics",
        },
        {
            "case": "Case 5",
            "comparison": "p-sensitivity + resolution dependence",
            "primary_metric": f"final -g_eq gain={p4_gain:.1f}% (p=4), {p6_gain:.1f}% (p=6)",
            "secondary_metric": (
                f"64^3 gap error={float(resolution['gap_error_p2_at_64']) * 100.0:.2f}% (p=2), "
                f"{float(resolution['gap_error_p6_at_64']) * 100.0:.1f}% (p=6)"
            ),
            "takeaway": "current numerical claim is scoped to p=2..4; p=6 is retained only as a trend demonstration",
        },
        {
            "case": "Case 6",
            "comparison": "two-contact normal-only consistency check",
            "primary_metric": (
                f"peak penetration={float(multicontact['peak_soccp_penetration']):.3e} / "
                f"{float(multicontact['peak_normal_lcp_penetration']):.3e} / "
                f"{float(multicontact['peak_penalty_penetration']):.3e}"
            ),
            "secondary_metric": (
                f"SOCCP-normal LCP peak diff={float(multicontact['soccp_vs_normal_lcp_peak_diff']):.2e}; "
                f"peak residuals={float(multicontact['peak_soccp_residual']):.2e}/{float(multicontact['peak_normal_lcp_residual']):.2e}"
            ),
            "takeaway": "the symmetric normal-only scene behaves as a solver-consistency check rather than a stress test",
        },
        {
            "case": "Case 7",
            "comparison": "controlled mass-ratio robustness sweep",
            "primary_metric": (
                f"mass ratio {int(round(float(mass_ratio['min_converged'])))} to "
                f"{int(round(float(mass_ratio['max_mass_ratio'])))} all converged"
            ),
            "secondary_metric": (
                f"max Newton iters={int(round(float(mass_ratio['max_iterations'])))}; "
                f"max residual inf-norm={float(mass_ratio['max_residual_inf']):.1e}"
            ),
            "takeaway": "the SOCCP core remains locally robust under controlled ill-conditioning",
        },
        {
            "case": "Case 8",
            "comparison": "3D tripod landing: SOCCP / full normal LCP / fixed-dt tuned penalty",
            "primary_metric": (
                f"peak penetration={float(tripod['peak_soccp_penetration']):.3e} / "
                f"{float(tripod['peak_normal_lcp_penetration']):.3e} / "
                f"{float(tripod['peak_penalty_penetration']):.3e}"
            ),
            "secondary_metric": (
                f"tuned penalty k/c={float(tripod_tuning['best_stiffness']):.0f}/"
                f"{float(tripod_tuning['best_damping']):.0f}; "
                f"final tilt={float(tripod['final_soccp_tilt_deg']):.1f} / "
                f"{float(tripod['final_normal_lcp_tilt_deg']):.1f} / "
                f"{float(tripod['final_penalty_tilt_deg']):.1f} deg"
            ),
            "takeaway": "NormalLCP restores the normal-only reference path; against the tuned penalty baseline, both complementarity solvers reduce penetration but do not dominate every stability metric",
        },
        {
            "case": "Case 9",
            "comparison": "oblique box 3D friction: SOCCP vs polyhedral external baseline",
            "primary_metric": (
                f"max slip/spin diff={float(oblique_poly['max_slip_diff']):.3e} / "
                f"{float(oblique_poly['max_spin_diff']):.3e}"
            ),
            "secondary_metric": (
                f"peak scaled residual={float(oblique_poly['peak_scaled_soccp']):.1e}/"
                f"{float(oblique_poly['peak_scaled_polyhedral']):.1e}; "
                f"max tilt diff={float(oblique_poly['max_tilt_diff']):.3e}"
            ),
            "takeaway": "the torsion-free 3D friction branch is no longer validated only against analytic sliding and PGS; it also remains close to an independent polyhedral-cone baseline in a nontrivial 3D scene",
        },
        {
            "case": "Case 10",
            "comparison": "oblique box on plane: 4D vs 3D vs tuned penalty",
            "primary_metric": (
                f"final spin={float(oblique['final_spin_4d']):.2e} / "
                f"{float(oblique['final_spin_3d']):.2f} / "
                f"{float(oblique['final_spin_penalty']):.2f} rad/s"
            ),
            "secondary_metric": (
                f"final slip={float(oblique['final_slip_4d']):.3f} / "
                f"{float(oblique['final_slip_3d']):.3f} / "
                f"{float(oblique['final_slip_penalty']):.3f} m"
            ),
            "takeaway": "4D friction changes full-engine 3D dynamics beyond the axisymmetric cylinder case, albeit with the hardest residual profile",
        },
        {
            "case": "Case 11",
            "comparison": "analytic box SDFs vs triangulated mesh-box SDFs in a two-contact stack",
            "primary_metric": (
                f"max upper-y diff={float(mesh_sdf['max_upper_y_diff']):.2e}; "
                f"max upper-vy diff={float(mesh_sdf['max_upper_vy_diff']):.2e}"
            ),
            "secondary_metric": (
                f"max penetration diff={float(mesh_sdf['max_penetration_diff']):.2e}; "
                f"peak scaled residual={float(mesh_sdf['peak_scaled_residual_analytic']):.1e}"
            ),
            "takeaway": "the current contact pipeline is no longer limited to analytical primitives and already remains consistent on a multi-contact scene with closed triangle-mesh SDF boxes",
        },
    ]

    with (RESULTS / "reviewer_comparison_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case", "comparison", "primary_metric", "secondary_metric", "takeaway"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    plot_cylinder_spin()
    plot_sharp_drop()
    plot_resolution_convergence()
    plot_mass_ratio_stack()
    plot_engine_multicontact()
    plot_engine_tripod_landing()
    plot_penalty_tripod_tuning()
    plot_engine_oblique_box_friction()
    plot_engine_oblique_box_polyhedral_compare()
    plot_engine_torsion_compare()
    plot_torsion_friction_reference()
    plot_4d_coupled_friction_reference()
    plot_sphere_drop_compare()
    plot_engine_sliding_friction_reference()
    plot_engine_mesh_sdf_multicontact()
    write_solver_convergence_summary()
    write_reviewer_comparison_summary()


if __name__ == "__main__":
    main()

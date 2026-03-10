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


def plot_engine_tripod_landing():
    rows = read_csv(RESULTS / "benchmark_engine_tripod_landing.csv")
    time = [float(row["time"]) for row in rows]
    soccp_pen = [float(row["soccp_penetration"]) for row in rows]
    penalty_pen = [float(row["penalty_penetration"]) for row in rows]
    soccp_tilt = [float(row["soccp_tilt_deg"]) for row in rows]
    penalty_tilt = [float(row["penalty_tilt_deg"]) for row in rows]
    soccp_contacts = [float(row["soccp_contacts"]) for row in rows]
    penalty_contacts = [float(row["penalty_contacts"]) for row in rows]

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.2), sharex=True)
    axes[0].plot(time, soccp_pen, linewidth=2.0, color="#0a9396", label="SOCCP engine")
    axes[0].plot(time, penalty_pen, linewidth=2.0, color="#ae2012", linestyle="--", label="penalty baseline")
    axes[0].set_ylabel("max penetration")
    axes[0].set_title("3D Tripod Landing Stress Test")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(time, soccp_tilt, linewidth=2.0, color="#005f73", label="SOCCP tilt")
    axes[1].plot(time, penalty_tilt, linewidth=2.0, color="#bb3e03", linestyle="--", label="penalty tilt")
    axes[1].set_ylabel("tilt angle (deg)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)

    axes[2].step(time, soccp_contacts, where="post", linewidth=2.0, color="#264653", label="SOCCP contacts")
    axes[2].step(time, penalty_contacts, where="post", linewidth=2.0, color="#9c6644", linestyle="--", label="penalty contacts")
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("active contacts")
    axes[2].set_yticks([0.0, 1.0, 2.0, 3.0])
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(RESULTS / "benchmark_engine_tripod_landing.png", dpi=180)
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


def write_reviewer_comparison_summary():
    cylinder = read_single_summary("benchmark_cylinder_spin_summary.csv")
    torsion = read_single_summary("benchmark_engine_torsion_compare_summary.csv")
    sharp_rows = read_csv(RESULTS / "benchmark_sharp_drop_summary.csv")
    sharp_by_p = {int(round(float(row["p_norm"]))): row for row in sharp_rows}
    resolution = read_single_summary("benchmark_resolution_convergence_summary.csv")
    multicontact = read_single_summary("benchmark_engine_multicontact_summary.csv")
    mass_ratio = read_single_summary("benchmark_mass_ratio_stack_summary.csv")
    tripod = read_single_summary("benchmark_engine_tripod_landing_summary.csv")

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
            "comparison": "full-engine 4D friction vs 3D torsion-free baseline",
            "primary_metric": (
                f"final spin={float(torsion['final_omega_4d']):.2e} vs "
                f"{float(torsion['final_omega_3d']):.2f} rad/s"
            ),
            "secondary_metric": f"residual spin reduction={float(torsion['spin_decay_ratio']) * 100.0:.3f}%",
            "takeaway": "torsional impulse changes the full-engine dynamics",
        },
        {
            "case": "Case 3",
            "comparison": "p-sensitivity + resolution dependence",
            "primary_metric": f"final -g_eq gain={p4_gain:.1f}% (p=4), {p6_gain:.1f}% (p=6)",
            "secondary_metric": (
                f"64^3 gap error={float(resolution['gap_error_p2_at_64']) * 100.0:.2f}% (p=2), "
                f"{float(resolution['gap_error_p6_at_64']) * 100.0:.1f}% (p=6)"
            ),
            "takeaway": "current numerical claim is scoped to p=2..4; p=6 is retained only as a trend demonstration",
        },
        {
            "case": "Case 4",
            "comparison": "full-engine SOCCP vs penalty baseline",
            "primary_metric": f"peak penetration reduced by {float(multicontact['penetration_reduction']) * 100.0:.1f}%",
            "secondary_metric": (
                f"peak residual={float(multicontact['peak_soccp_residual']):.2e}; "
                f"max Newton iters={int(round(float(multicontact['max_soccp_iterations'])))}"
            ),
            "takeaway": "SOCCP reduces penetration with stable iterative cost on the two-contact scene",
        },
        {
            "case": "Case 5",
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
            "case": "Case 6",
            "comparison": "3D tripod landing: SOCCP vs penalty baseline",
            "primary_metric": f"peak penetration reduced by {float(tripod['penetration_reduction']) * 100.0:.1f}%",
            "secondary_metric": (
                f"final tilt={float(tripod['final_soccp_tilt_deg']):.1f} vs "
                f"{float(tripod['final_penalty_tilt_deg']):.1f} deg"
            ),
            "takeaway": "SOCCP maintains support stability in a stronger 3D multi-contact scene",
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
    plot_engine_torsion_compare()
    plot_sphere_drop_compare()
    write_reviewer_comparison_summary()


if __name__ == "__main__":
    main()

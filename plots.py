# Importing the libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import N_PATHS

from simulate import (
    run_multi_tranche_mc,
    prepare_static_inputs,
    simulate_hull_white_paths,
    predict_cpr_with_rates,
)

# Utility
def ensure_plot_dir():
    if not os.path.exists("plots"):
        os.makedirs("plots")

def save_fig(name):
    """Save plot only; show at end."""
    ensure_plot_dir()
    plt.tight_layout()
    plt.savefig(f"plots/{name}.png", dpi=180)

# MAIN PLOTTING SCRIPT
def main():
    print("Running Monte Carlo + generating plots\n")

    # 1. Prepare static inputs
    static = prepare_static_inputs()
    n_steps = static["n_steps"]
    tranches = static["tranches"]
    cpr_model = static["cpr_model"]
    mortgage_rate = static["mortgage_rate"]

    tranche_names = [t.name for t in tranches]
    age_months = np.arange(n_steps)

    # 2. Generate base short rate paths
    np.random.seed(42)
    r_paths_base = simulate_hull_white_paths(N_PATHS, n_steps)

    # (A) PLOT: Hull–White Short Rate Sample Paths
    plt.figure(figsize=(10, 5))
    plt.plot(r_paths_base[:20].T, alpha=0.6)
    plt.title("Hull–White Short Rate Sample Paths (20 drawn)")
    plt.xlabel("Month")
    plt.ylabel("Short Rate")
    save_fig("rate_paths")

    # (B) PLOT: Example CPR Path
    example_r = r_paths_base[0]
    cpr_example = predict_cpr_with_rates(
        age_months=age_months,
        model=cpr_model,
        r_path=example_r,
        mortgage_rate=mortgage_rate,
    )

    plt.figure(figsize=(10, 4))
    plt.plot(cpr_example, lw=2)
    plt.title("Example CPR Path")
    plt.xlabel("Month")
    plt.ylabel("CPR Level")
    save_fig("cpr_example")

    # (C) RUN FULL MULTI-TRANCHE MC
    all_results = run_multi_tranche_mc()

    # Helper to get values even if missing
    def get_result(d, tranche, field):
        if tranche in d and field in d[tranche]:
            v = d[tranche][field]
            return 0 if v is None or np.isnan(v) else v
        return 0

    # (D) Base scenario mean prices
    base = all_results["Base"]
    prices = [get_result(base, t, "mean_price") for t in tranche_names]

    plt.figure(figsize=(8, 4))
    plt.bar(tranche_names, prices)
    plt.title("Tranche Mean Prices – Base Scenario")
    plt.ylabel("Price")
    save_fig("base_mean_prices")

    # (F) Scenario comparison – mean prices
    scen_names = list(all_results.keys())
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.15
    x = np.arange(len(tranche_names))

    for i, scen in enumerate(scen_names):
        scen_prices = [
            get_result(all_results[scen], t, "mean_price") for t in tranche_names
        ]
        ax.bar(x + i * width, scen_prices, width, label=scen)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(tranche_names)
    ax.set_title("Mean Tranche Prices Under Different Scenarios")
    ax.set_ylabel("Price")
    ax.legend()
    save_fig("scenario_mean_prices")

    # (G) WAL per scenario
    fig, ax = plt.subplots(figsize=(12, 5))

    # collect WALs by scenario first
    scen_wals = {}
    for scen in scen_names:
        scen_wals[scen] = [
            get_result(all_results[scen], t, "wal") for t in tranche_names
        ]

    # keep only tranches that ever have WAL > 0  
    has_wal = []
    for j, t in enumerate(tranche_names):
        if any(scen_wals[scen][j] > 0 for scen in scen_names):
            has_wal.append(j)

    # Filtered labels and positions
    wal_tranche_names = [tranche_names[j] for j in has_wal]
    x = np.arange(len(wal_tranche_names))
    width = 0.15

    for i, scen in enumerate(scen_names):
        vals = [scen_wals[scen][j] for j in has_wal]
        ax.bar(x + i * width, vals, width, label=scen)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(wal_tranche_names)
    ax.set_title("Weighted-Average Life (WAL) by Scenario")
    ax.set_ylabel("Years")
    ax.legend()
    save_fig("scenario_wal")

    # (H) Duration & Convexity
    base_res = all_results["Base"]
    up_res = all_results["+100bp"]
    down_res = all_results["-100bp"]

    dr = 0.01

    duration_rows = []
    convex_rows = []

    for t in tranche_names:
        P0 = get_result(base_res, t, "mean_price")
        Pup = get_result(up_res, t, "mean_price")
        Pdn = get_result(down_res, t, "mean_price")

        duration = -(Pup - Pdn) / (2 * P0 * dr) if P0 != 0 else 0
        convexity = (Pup + Pdn - 2 * P0) / (P0 * dr**2) if P0 != 0 else 0

        duration_rows.append([t, duration])
        convex_rows.append([t, convexity])

    df_dur = pd.DataFrame(duration_rows, columns=["Tranche", "Duration"])
    df_con = pd.DataFrame(convex_rows, columns=["Tranche", "Convexity"])

    print("\nDuration Table:\n", df_dur)
    print("\nConvexity Table:\n", df_con)

    # Duration plot
    plt.figure(figsize=(9, 4))
    plt.bar(df_dur["Tranche"], df_dur["Duration"])
    plt.title("Duration (Approx) – Parallel 100bp Shock")
    plt.ylabel("Duration")
    save_fig("duration_plot")

    # Convexity plot
    plt.figure(figsize=(9, 4))
    plt.bar(df_con["Tranche"], df_con["Convexity"])
    plt.title("Convexity (Approx) – Parallel 100bp Shock")
    plt.ylabel("Convexity")
    save_fig("convexity_plot")

    print("\n✓ All plots saved to: /plots directory")
    print("✓ Script finished. Windows will remain open.\n")

    plt.show()
    
if __name__ == "__main__":
    main()

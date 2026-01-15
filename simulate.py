import numpy as np
import pandas as pd

from config import N_PATHS, DT_MONTHS, HORIZON_YEARS
from rates import (
    discount_factors_from_short_rate_path,
    simulate_hull_white_paths,
)
from cmo_structure import build_tranches_from_deal
from waterfall import generate_collateral_path, allocate_multi_tranche
from data_prepay import (
    load_deal_data,
    compute_pool_cpr,
    fit_cpr_model,
)

# Import the whole module so we can check what functions exist
import data_prepay as _dp

# CPR prediction wrapper  
if hasattr(_dp, "predict_cpr_with_rates"):

    def predict_cpr_with_rates(age_months,
                               model,
                               r_path,
                               mortgage_rate):
        """
        Thin wrapper around data_prepay.predict_cpr_with_rates.

        If your data_prepay.py has the enhanced rate-sensitive CPR model,
        this will call it with the appropriate arguments.
        """
        return _dp.predict_cpr_with_rates(
            age_months=age_months,
            model=model,
            r_path=r_path,
            mortgage_rate=mortgage_rate,
        )
else:
    
    def predict_cpr_with_rates(age_months,
                               model,
                               r_path,
                               mortgage_rate):
        """
        Fallback: ignore rates and just call predict_cpr_from_age.
        This keeps the rest of the Monte Carlo code working even if
        you didn't update data_prepay.py yet.
        """
        return _dp.predict_cpr_from_age(age_months, model)

# WAL helper
def compute_wal_from_principal_paths(principal_paths: np.ndarray,
                                     dt_months: int = DT_MONTHS) -> float:
    """
    Compute Weighted Average Life (WAL) in years from a matrix of principal
    payments across Monte Carlo paths.

    Parameters
    ----------
    principal_paths : np.ndarray
        Shape (n_paths, n_steps). Each row is a principal schedule for one path.
    dt_months : int
        Time step size in months (default: DT_MONTHS, usually 1).

    Returns
    -------
    float
        WAL in years, based on expected principal over paths.
    """
    if principal_paths.size == 0:
        return 0.0

    n_paths, n_steps = principal_paths.shape

    # Expected principal per month
    expected_prin = principal_paths.mean(axis=0)   
    total_prin = expected_prin.sum()

    if total_prin <= 1e-8:
        return 0.0

    # Payment times in years (month-end)
    dt_years = dt_months / 12.0
    # Month index 0 -> time dt_years, index 1 -> 2*dt_years, etc.
    times_years = (np.arange(n_steps) + 1) * dt_years

    wal_years = np.sum(expected_prin * times_years) / total_prin
    return wal_years

# Static prep: load deal, fit CPR model, build tranches, etc.
def prepare_static_inputs():
    """
    Load all static objects needed for Monte Carlo and scenarios:
    - deal data
    - pool CPR history + CPR model
    - tranche structure
    - initial pool balance
    - mortgage rate
    - time horizon (n_steps)
    """
    # 1. Load deal + CPR  
    deal_data = load_deal_data()
    collateral_df = deal_data["collateral_all"]

    pool_cpr_df = compute_pool_cpr(collateral_df)

    cpr_path_hist = pool_cpr_df["cpr"].reset_index(drop=True)
    n_hist = len(cpr_path_hist)

    mean_cpr = cpr_path_hist.mean()
    print(f"Mean pool CPR (historical): {mean_cpr:.4%}")
    print(f"Number of months of history: {n_hist}")

    # 2. Fit CPR model on history  
    cpr_model = fit_cpr_model(pool_cpr_df)
    print("Fitted CPR model beta:", cpr_model.get("beta"))

    sigma_eps = cpr_model.get("sigma_eps", None)
    if sigma_eps is not None:
        print("Residual CPR std (sigma_eps):", sigma_eps)

    first_month = cpr_model.get("first_month", None)
    if first_month is not None:
        print("First month in history:", first_month)

    # 3. Build all tranches  
    tranches = build_tranches_from_deal(deal_data)
    print("Tranches loaded:")
    for tr in tranches:
        print(f"  {tr.name}: balance={tr.initial_balance:.2f}, "
              f"coupon={tr.coupon:.4f}, type={tr.tranche_type}, "
              f"priority={tr.priority}")

    if not tranches:
        raise RuntimeError("No tranches were built from the deal data!")

    # 4. Simulation config  
    max_steps_from_horizon = int(HORIZON_YEARS * 12 / DT_MONTHS)
    n_steps = min(max_steps_from_horizon, n_hist)
    print("Using n_steps =", n_steps)

    initial_balance = float(collateral_df.iloc[0]["balance"])

    # Approximate pool mortgage rate: take the senior tranche coupon + small spread
    senior_tranche = tranches[0]
    mortgage_rate = senior_tranche.coupon + 0.005
    print(f"Using mortgage_rate = {mortgage_rate:.4f} "
          f"(senior coupon {senior_tranche.coupon:.4f} + 50bp)")

    static = {
        "deal_data": deal_data,
        "collateral_df": collateral_df,
        "pool_cpr_df": pool_cpr_df,
        "cpr_model": cpr_model,
        "tranches": tranches,
        "initial_balance": initial_balance,
        "mortgage_rate": mortgage_rate,
        "n_steps": n_steps,
        "n_hist": n_hist,
    }

    return static

# Core scenario engine
def run_single_scenario(static_inputs,
                        r_paths_base: np.ndarray,
                        scenario_name: str,
                        curve_shift_bp: float = 0.0,
                        cpr_scale: float = 1.0) -> dict:
    """
    Run the full multi-tranche Monte Carlo for a single scenario.

    Parameters
    ----------
    static_inputs : dict
        Output of prepare_static_inputs().
    r_paths_base : np.ndarray
        Shape (n_paths, n_steps) base Hull–White short-rate paths.
    scenario_name : str
        A label for printing.
    curve_shift_bp : float
        Parallel shift of short-rate paths in basis points.
        E.g. +100 means r_t -> r_t + 0.01.
    cpr_scale : float
        Global multiplicative factor on the modeled CPR (e.g. 1.3 for 130%).

    Returns
    -------
    dict
        scenario_results[tranche_name] = {
            "mean_price": ...,
            "std_price": ...,
            "min_price": ...,
            "max_price": ...,
            "wal": ...,
        }
    """
    tranches = static_inputs["tranches"]
    cpr_model = static_inputs["cpr_model"]
    initial_balance = static_inputs["initial_balance"]
    mortgage_rate = static_inputs["mortgage_rate"]
    n_steps = static_inputs["n_steps"]

    n_paths, n_steps_r = r_paths_base.shape
    assert n_steps_r >= n_steps, "r_paths_base must have at least n_steps columns"

    age_months = np.arange(n_steps)
    shift_dec = curve_shift_bp / 10000.0  # bp -> decimal

    # prices[tranche_name] = list of PVs (one per path)
    prices_dict = {tr.name: [] for tr in tranches}
    # principal_paths[tranche_name] = list of principal arrays (one per path)
    principal_paths: dict[str, list[np.ndarray]] = {tr.name: [] for tr in tranches}

    # Monte Carlo loop
    for p in range(n_paths):
        # Apply parallel rate shift to base path
        r_path_scen = r_paths_base[p, :n_steps] + shift_dec

        # CPR path, rate-sensitive if enhanced function exists,
        cpr_pred = predict_cpr_with_rates(
            age_months=age_months,
            model=cpr_model,
            r_path=r_path_scen,
            mortgage_rate=mortgage_rate,
        )

        # Global CPR scale stress (e.g. 0.7x, 1.3x)
        cpr_pred = np.clip(cpr_pred * cpr_scale, 0.0, 1.0)
        cpr_path = pd.Series(cpr_pred)

        # Collateral cashflows for this scenario
        coll_cf = generate_collateral_path(
            initial_balance=initial_balance,
            mortgage_rate=mortgage_rate,
            cpr_path=cpr_path,
            n_months=n_steps,
        )

        # Tranche cashflows under multi-tranche waterfall
        tranche_cfs = allocate_multi_tranche(coll_cf, tranches)

        # Discount factors from r_path for this scenario
        df = discount_factors_from_short_rate_path(r_path_scen, dt_months=DT_MONTHS)
        df = df[:n_steps]

        # Present values + store principal schedules per tranche on this path
        for tr in tranches:
            name = tr.name
            cf_df = tranche_cfs[name]
            cf_t = cf_df["tranche_cashflow"].values
            prin_t = cf_df["tranche_principal"].values

            pv = np.sum(cf_t * df)
            prices_dict[name].append(pv)
            principal_paths[name].append(prin_t)

    # Summarizing the results
    print()
    print(f"Scenario: {scenario_name}")
    scenario_results: dict[str, dict] = {}

    for tr in tranches:
        name = tr.name
        prices = np.array(prices_dict[name])

        if len(principal_paths[name]) > 0:
            prin_mat = np.vstack(principal_paths[name])
            wal_years = compute_wal_from_principal_paths(prin_mat, dt_months=DT_MONTHS)
        else:
            wal_years = 0.0

        mean_price = prices.mean()
        std_price = prices.std()
        min_price = prices.min()
        max_price = prices.max()

        print(f"Tranche {name}:")
        print(f"  Mean price: {mean_price:,.2f}")
        print(f"  Std dev:    {std_price:,.2f}")
        print(f"  Min / Max:  {min_price:,.2f}  /  {max_price:,.2f}")
        print(f"  WAL:        {wal_years:.2f} years")
        print()

        scenario_results[name] = {
            "mean_price": float(mean_price),
            "std_price": float(std_price),
            "min_price": float(min_price),
            "max_price": float(max_price),
            "wal": float(wal_years),
        }

    return scenario_results

# Main driver: run base + stress scenarios and compute duration/convexity
def run_multi_tranche_mc():
    # 1) Static prep
    static_inputs = prepare_static_inputs()
    n_steps = static_inputs["n_steps"]

    # 2) Generate base Hull–White short-rate paths once,
    #    then use them for all scenarios with different shifts.
    np.random.seed(42)  # for reproducibility
    r_paths_base = simulate_hull_white_paths(N_PATHS, n_steps)

    # 3) Define scenarios
    scenarios = [
        {"name": "Base",            "curve_shift_bp":    0.0, "cpr_scale": 1.0},
        {"name": "+100bp",          "curve_shift_bp":  100.0, "cpr_scale": 1.0},
        {"name": "-100bp",          "curve_shift_bp": -100.0, "cpr_scale": 1.0},
        {"name": "High CPR (x1.3)", "curve_shift_bp":    0.0, "cpr_scale": 1.3},
        {"name": "Low CPR (x0.7)",  "curve_shift_bp":    0.0, "cpr_scale": 0.7},
    ]

    all_results: dict[str, dict[str, dict]] = {}

    # 4) Running the scenarios
    for scen in scenarios:
        res = run_single_scenario(
            static_inputs=static_inputs,
            r_paths_base=r_paths_base,
            scenario_name=scen["name"],
            curve_shift_bp=scen["curve_shift_bp"],
            cpr_scale=scen["cpr_scale"],
        )
        all_results[scen["name"]] = res

    # 5) Approximate duration & convexity from ±100bp shocks
    if {"Base", "+100bp", "-100bp"}.issubset(all_results.keys()):
        print("\nApproximate parallel-rate duration and convexity "
              "(using ±100bp short-rate shocks):")

        base_res = all_results["Base"]
        up_res = all_results["+100bp"]
        down_res = all_results["-100bp"]

        dr = 0.01  # 100bp in decimal

        for tr in static_inputs["tranches"]:
            name = tr.name
            P0 = base_res[name]["mean_price"]
            P_up = up_res[name]["mean_price"]
            P_down = down_res[name]["mean_price"]

            # Numerical duration and convexity
            duration = -(P_up - P_down) / (2.0 * P0 * dr)
            convexity = (P_up + P_down - 2.0 * P0) / (P0 * dr * dr)

            print(f"Tranche {name}:")
            print(f"  Duration (approx):  {duration:,.3f}")
            print(f"  Convexity (approx): {convexity:,.3f}")
            print()

    return all_results

if __name__ == "__main__":
    run_multi_tranche_mc()
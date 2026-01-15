import numpy as np
import pandas as pd
from typing import Dict, List
from cmo_structure import Tranche


# Collateral engine 
def generate_collateral_path(initial_balance: float,
                             mortgage_rate: float,
                             cpr_path: pd.Series,
                             n_months: int) -> pd.DataFrame:
    """
    Generate collateral (pool) cashflows for a single scenario/path.

    We assume:
    - Level-payment mortgage structure (fixed mortgage_rate)
    - CPR_path gives the annualized CPR for each month
    - Prepayments follow standard SMM formula:
        SMM_t = 1 - (1 - CPR_t)**(1/12)
    - Scheduled principal + prepayments + interest are computed monthly.
    """
    assert len(cpr_path) >= n_months, "CPR path must have at least n_months entries."

    r_m = mortgage_rate / 12.0

    if r_m > 1e-8:
        payment = initial_balance * r_m / (1.0 - (1.0 + r_m) ** (-n_months))
    else:
        payment = initial_balance / n_months

    balances = np.zeros(n_months + 1)
    balances[0] = initial_balance

    sched_prin_arr = np.zeros(n_months)
    prepay_arr = np.zeros(n_months)
    interest_arr = np.zeros(n_months)
    total_prin_arr = np.zeros(n_months)

    for t in range(n_months):
        balance_prev = balances[t]

        if balance_prev <= 1e-8:
            balances[t+1:] = 0.0
            break

        interest_t = balance_prev * r_m

        sched_prin_t = max(payment - interest_t, 0.0)
        sched_prin_t = min(sched_prin_t, balance_prev)

        cpr_t = float(cpr_path.iloc[t])
        smm_t = 1.0 - (1.0 - cpr_t) ** (1.0 / 12.0)

        prepay_base = balance_prev - sched_prin_t
        prepay_t = smm_t * max(prepay_base, 0.0)
        prepay_t = min(prepay_t, balance_prev - sched_prin_t)

        total_prin_t = sched_prin_t + prepay_t
        balance_new = balance_prev - total_prin_t

        sched_prin_arr[t] = sched_prin_t
        prepay_arr[t] = prepay_t
        interest_arr[t] = interest_t
        total_prin_arr[t] = total_prin_t
        balances[t + 1] = balance_new

    df = pd.DataFrame({
        "month_idx": np.arange(n_months),
        "balance": balances[:-1],
        "sched_prin": sched_prin_arr,
        "prepay": prepay_arr,
        "total_prin": total_prin_arr,
        "interest": interest_arr,
    })

    return df

#  Single-tranche waterfall  
def allocate_single_tranche(collateral_cf: pd.DataFrame,
                            tranche: Tranche) -> pd.DataFrame:
    """
    Allocate collateral cashflows to a single tranche.
    """
    n_months = len(collateral_cf)
    tranche_balances = np.zeros(n_months + 1)
    tranche_balances[0] = tranche.initial_balance

    tranche_prin_arr = np.zeros(n_months)
    tranche_int_arr = np.zeros(n_months)
    tranche_cf_arr = np.zeros(n_months)

    r_tranche_m = tranche.coupon / 12.0

    for t in range(n_months):
        bal_prev = tranche_balances[t]
        if bal_prev <= 1e-8:
            tranche_balances[t+1:] = 0.0
            break

        interest_due = bal_prev * r_tranche_m
        tranche_int = interest_due

        coll_prin_t = float(collateral_cf.loc[t, "total_prin"])
        tranche_prin = min(coll_prin_t, bal_prev)

        cf_t = tranche_int + tranche_prin
        bal_new = bal_prev - tranche_prin

        tranche_prin_arr[t] = tranche_prin
        tranche_int_arr[t] = tranche_int
        tranche_cf_arr[t] = cf_t
        tranche_balances[t + 1] = bal_new

    df_tranche = pd.DataFrame({
        "month_idx": np.arange(n_months),
        "tranche_balance": tranche_balances[:-1],
        "tranche_principal": tranche_prin_arr,
        "tranche_interest": tranche_int_arr,
        "tranche_cashflow": tranche_cf_arr,
    })

    return df_tranche

# Multi-tranche waterfall 
def allocate_multi_tranche(collateral_cf: pd.DataFrame,
                           tranches: List[Tranche]) -> Dict[str, pd.DataFrame]:
    """
    Allocate collateral cashflows across multiple tranches using a simple
    sequential principal waterfall.

    Assumptions (v1):
    - Each tranche earns coupon * starting balance / 12 as interest (no shortfall).
    - Principal is paid SEQUENTIALLY by priority:
        * All collateral principal goes first to priority 1
        * When that tranche is paid off, principal flows to priority 2, etc.

    Parameters
    ----------
    collateral_cf : DataFrame
        Output of generate_collateral_path, with columns:
          - month_idx
          - total_prin
          - interest   (not strictly used here, but available)
    tranches : list of Tranche
        All tranches to include, with valid priority fields.

    Returns
    -------
    dict
        Mapping tranche.name -> DataFrame with columns:
          - month_idx
          - tranche_balance
          - tranche_principal
          - tranche_interest
          - tranche_cashflow
    """
    n_months = len(collateral_cf)

    # Sort tranches by priority: lower = more senior
    tranches_sorted = sorted(tranches, key=lambda tr: tr.priority)
    n_tr = len(tranches_sorted)

    # Initialize storage per tranche
    balances = {
        tr.name: np.zeros(n_months + 1) for tr in tranches_sorted
    }
    prin = {
        tr.name: np.zeros(n_months) for tr in tranches_sorted
    }
    intr = {
        tr.name: np.zeros(n_months) for tr in tranches_sorted
    }
    cf = {
        tr.name: np.zeros(n_months) for tr in tranches_sorted
    }

    # Set initial balances
    for tr in tranches_sorted:
        balances[tr.name][0] = tr.initial_balance

    # Monthly coupons
    monthly_coupons = {
        tr.name: tr.coupon / 12.0 for tr in tranches_sorted
    }

    # Loop over months
    for t in range(n_months):
        coll_prin_t = float(collateral_cf.loc[t, "total_prin"])

        # First compute interest for all tranches (no shortfall logic yet)
        for tr in tranches_sorted:
            name = tr.name
            bal_prev = balances[name][t]
            if bal_prev <= 1e-8:
                intr[name][t] = 0.0
            else:
                intr[name][t] = bal_prev * monthly_coupons[name]

        # Now allocate principal sequentially by priority
        principal_remaining = coll_prin_t

        for tr in tranches_sorted:
            name = tr.name
            bal_prev = balances[name][t]

            if bal_prev <= 1e-8 or principal_remaining <= 1e-8:
                prin[name][t] = 0.0
                balances[name][t+1] = bal_prev
                continue

            tranche_prin = min(bal_prev, principal_remaining)
            prin[name][t] = tranche_prin
            balances[name][t+1] = bal_prev - tranche_prin
            principal_remaining -= tranche_prin

        # Update cashflows per tranche
        for tr in tranches_sorted:
            name = tr.name
            cf[name][t] = prin[name][t] + intr[name][t]

        # Any leftover principal (if principal_remaining > 0) is ignored for v1.

    # Build output DataFrames per tranche
    out: Dict[str, pd.DataFrame] = {}
    month_idx = np.arange(n_months)

    for tr in tranches_sorted:
        name = tr.name
        out[name] = pd.DataFrame({
            "month_idx": month_idx,
            "tranche_balance": balances[name][:-1],
            "tranche_principal": prin[name],
            "tranche_interest": intr[name],
            "tranche_cashflow": cf[name],
        })

    return out
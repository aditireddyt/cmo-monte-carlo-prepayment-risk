# Importing the libraries
import numpy as np
import pandas as pd
from typing import Dict
from config import DEAL_FILE

def load_deal_data() -> Dict[str, pd.DataFrame]:
    """
    Load the core Freddie Mac REMIC deal Excel file.

    Returns
    -------
    dict
        Keys:
          - 'collateral_all': 'Collateral all' sheet
          - 'collateral_4' : 'Collateral 4' sheet
          - 'all_bonds'    : 'All Bonds' sheet
          - 'bond_PA', 'bond_PI', 'bond_PL', 'bond_W',
            'bond_WA', 'bond_WB', 'bond_WC' : individual bond sheets
    """
    xls = pd.ExcelFile(DEAL_FILE)
    data: Dict[str, pd.DataFrame] = {}

    # Core collateral and bonds
    if "Collateral all" in xls.sheet_names:
        data["collateral_all"] = xls.parse("Collateral all")
    if "Collateral 4" in xls.sheet_names:
        data["collateral_4"] = xls.parse("Collateral 4")
    if "All Bonds" in xls.sheet_names:
        data["all_bonds"] = xls.parse("All Bonds")

    # Individual bond / tranche sheets
    bond_sheet_map = {
        "PA": "Bond PA",
        "PI": "Bond PI",
        "PL": "Bond PL",
        "W":  "Bond W",
        "WA": "Bond WA",
        "WB": "Bond WB",
        "WC": "Bond WC",
    }

    for short_name, sheet_name in bond_sheet_map.items():
        if sheet_name in xls.sheet_names:
            key = f"bond_{short_name}"
            data[key] = xls.parse(sheet_name)

    return data

def compute_pool_cpr(collateral_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly SMM and CPR for the pool using the 'Collateral all' sheet.

    Parameters
    ----------
    collateral_df : DataFrame
        Expected columns:
        - 'Date'
        - 'balance'  : end-of-month balance
        - 'sched'    : scheduled principal
        - 'unsched'  : unscheduled principal (prepayments)

    Returns
    -------
    DataFrame
        Columns:
        - date
        - balance
        - sched
        - unsched
        - smm
        - cpr
        - age_months
        - month
        - year
    """
    df = collateral_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # Renaming the columns for the ease
    df = df.rename(columns={
        "Date": "date",
        "balance": "balance",
        "sched": "sched",
        "unsched": "unsched",
    })

    # Avoid division by zero: if balance + unsched = 0, SMM = 0
    denom = df["balance"] + df["unsched"]
    with np.errstate(divide="ignore", invalid="ignore"):
        smm = df["unsched"] / denom.replace(0, np.nan)
    smm = smm.fillna(0.0)

    # Monthly SMM and annualized CPR
    df["smm"] = smm
    df["cpr"] = 1.0 - (1.0 - df["smm"]) ** 12

    # Basic time features
    df = df.sort_values("date").reset_index(drop=True)
    df["age_months"] = np.arange(len(df))
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year

    return df[[
        "date", "balance", "sched", "unsched",
        "smm", "cpr", "age_months", "month", "year"
    ]]


def fit_cpr_model(pool_cpr_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Fit a richer CPR model on the pool history.

    Base specification:
        CPR_t ≈ beta0 + beta1 * age_t + beta2 * age_t^2

    Plus:
      - Seasonality factors by calendar month (Jan..Dec)
      - Residual volatility (for adding noise in simulation)
      - First calendar month (to align age -> month in simulation)

    NOTE: We do NOT regress on rates here because we don't have
    historical rate data aligned with this deal. Rate sensitivity
    is added in simulation using a proxy incentive term.

    Returns
    -------
    dict
        Model object containing:
          - 'beta' : np.ndarray of shape (3,)
          - 'sigma_eps' : float, residual std from base regression
          - 'seasonality_factors' : np.ndarray of shape (12,)
          - 'global_mean_cpr' : float
          - 'first_month' : int in [1, 12]
    """
    df = pool_cpr_df.copy().dropna(subset=["cpr", "age_months", "month"])

    y = df["cpr"].values  # shape (n,)
    age = df["age_months"].values.astype(float)

    # Design matrix X: [1, age, age^2]
    X = np.column_stack([
        np.ones_like(age),
        age,
        age ** 2,
    ])

    # Least squares solution: beta = (X'X)^(-1) X'y
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    # Residuals & their std (for stochastic CPR component)
    y_hat = X @ beta
    resid = y - y_hat
    
    # ddof=1 for unbiased sample std; guard against very small sample sizes
    sigma_eps = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0

    # Seasonality: average CPR by calendar month vs global mean
    global_mean = float(df["cpr"].mean()) if len(df) > 0 else 0.0
    month_means = df.groupby("month")["cpr"].mean()

    seasonality = np.ones(12, dtype=float)
    if global_mean > 0:
        for m in range(1, 13):
            if m in month_means.index:
                factor = float(month_means.loc[m] / global_mean)
                # Avoid extreme factors
                seasonality[m - 1] = np.clip(factor, 0.5, 1.5)
            else:
                seasonality[m - 1] = 1.0
    else:
        seasonality[:] = 1.0

    # First calendar month in the history (to align age -> month)
    first_month = int(df.sort_values("date").iloc[0]["month"]) if len(df) > 0 else 1

    model = {
        "beta": beta,
        "sigma_eps": sigma_eps,
        "seasonality_factors": seasonality,
        "global_mean_cpr": global_mean,
        "first_month": first_month,
    }
    return model


def predict_cpr_from_age(age_months: np.ndarray,
                         model: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Backwards-compatible helper: predict CPR given age_in_months
    using ONLY the polynomial on age (no seasonality, no rates).

    Parameters
    ----------
    age_months : np.ndarray
        1D array of ages in months, shape (n_steps,)
    model : dict
        Model object returned by fit_cpr_model.

    Returns
    -------
    np.ndarray
        Predicted CPR values, clipped to [0, 1].
    """
    beta = model["beta"]
    age = age_months.astype(float)

    X = np.column_stack([
        np.ones_like(age),
        age,
        age ** 2,
    ])

    cpr_pred = X @ beta
    cpr_pred = np.clip(cpr_pred, 0.0, 1.0)
    return cpr_pred

def _predict_cpr_age_season(age_months: np.ndarray,
                            start_month: int,
                            model: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Internal helper: base CPR as function of age + seasonality.

    Parameters
    ----------
    age_months : np.ndarray
        1D array of ages in months.
    start_month : int
        Calendar month (1..12) corresponding to age_months == 0
        in the simulation.
    model : dict
        Model from fit_cpr_model().
    """
    beta = model["beta"]
    seasonality = model.get("seasonality_factors", np.ones(12, dtype=float))

    age = age_months.astype(float)
    X = np.column_stack([
        np.ones_like(age),
        age,
        age ** 2,
    ])

    cpr_base = X @ beta  # base effect from age
    cpr_base = np.clip(cpr_base, 0.0, 1.0)

    # Align month index: 0 -> January, ..., 11 -> December
    start_idx = int(start_month) - 1  # 0..11
    # For each age t, month index is (start_idx + t) mod 12
    month_idx = (start_idx + age_months.astype(int)) % 12
    month_factors = seasonality[month_idx]

    cpr_season = cpr_base * month_factors
    cpr_season = np.clip(cpr_season, 0.0, 1.0)
    return cpr_season

def generate_cpr_path(age_months: np.ndarray,
                      mortgage_rate: float,
                      r_path: np.ndarray,
                      cpr_model: Dict[str, np.ndarray],
                      start_month: int | None = None,
                      rng: np.random.Generator | None = None,
                      refi_sensitivity: float = 4.0,
                      mortgage_spread: float = 0.015,
                      noise_scale: float = 1.0) -> np.ndarray:
    """
    Generate a CPR path for one Monte Carlo scenario, with:
      - baseline age effect
      - seasonality by calendar month
      - refi incentive based on rate path
      - stochastic noise from regression residuals

    Parameters
    ----------
    age_months : np.ndarray
        1D array of ages in months, length = n_steps.
    mortgage_rate : float
        Pool mortgage rate (annual, decimal), e.g. 0.05.
    r_path : np.ndarray
        Short-rate path (annual, decimal), shape (n_steps,).
        We use this as a proxy to derive a "market mortgage rate".
    cpr_model : dict
        Object returned by fit_cpr_model().
    start_month : int, optional
        Calendar month (1..12) for age 0. If None, use model['first_month'].
    rng : np.random.Generator, optional
        Random generator. If None, a new default_rng() is created.
    refi_sensitivity : float
        Strength of refi incentive effect. Higher => more CPR reaction
        when market rates fall below mortgage_rate.
    mortgage_spread : float
        Constant spread added to r_path to approximate a market
        mortgage rate: r_mkt_t ≈ r_path_t + mortgage_spread.
    noise_scale : float
        Multiplier on the residual std 'sigma_eps' for CPR noise.

    Returns
    -------
    np.ndarray
        CPR path, shape (n_steps,), clipped to [0, 1].
    """
    age_months = np.asarray(age_months, dtype=float)
    n_steps = age_months.shape[0]

    r_path = np.asarray(r_path, dtype=float)
    if r_path.shape[0] < n_steps:
        raise ValueError("r_path must have at least n_steps entries.")

    if start_month is None:
        start_month = int(cpr_model.get("first_month", 1))

    # 1) Base CPR from age + seasonality
    cpr_det = _predict_cpr_age_season(age_months, start_month, cpr_model)

    # 2) Refi incentive term: mortgage_rate vs "market mortgage rate"
    # Approximate market mortgage rate as short-rate + spread
    market_mort_rate = r_path[:n_steps] + mortgage_spread
    incentive = np.maximum(mortgage_rate - market_mort_rate, 0.0)

    # Linear scaling of CPR by (1 + refi_sensitivity * incentive)
    cpr_refi = cpr_det * (1.0 + refi_sensitivity * incentive)

    # 3) Add stochastic noise using residual std from regression
    sigma_eps = float(cpr_model.get("sigma_eps", 0.0)) * float(noise_scale)
    if rng is None:
        rng = np.random.default_rng()

    if sigma_eps > 1e-8:
        noise = rng.normal(loc=0.0, scale=sigma_eps, size=n_steps)
    else:
        noise = 0.0
        
    cpr_total = cpr_refi + noise
    cpr_total = np.clip(cpr_total, 0.0, 1.0)
    return cpr_total
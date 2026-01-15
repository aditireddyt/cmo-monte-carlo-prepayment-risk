import numpy as np
from typing import Dict, Tuple
from config import DT_MONTHS, HORIZON_YEARS, BASE_SHORT_RATE, HW_A, HW_SIGMA
from curve_loader import build_zero_curve_from_sofr, interpolate_zero_rate

# Build zero curve
def build_zero_curve() -> Dict[float, float]:
    """
    Build the zero curve used for rate modeling and discounting.

    For now, this is based on the USD SOFR curve exported from Bloomberg.
    If anything fails, you can fall back to a flat curve using BASE_SHORT_RATE.
    """
    zero_curve = build_zero_curve_from_sofr()
    if not zero_curve:
        # Fallback: simple flat curve
        return {
            1.0: BASE_SHORT_RATE,
            5.0: BASE_SHORT_RATE,
            10.0: BASE_SHORT_RATE,
        }
    return zero_curve

# Constant-rate paths
def generate_constant_rate_paths(n_paths: int,
                                 n_steps: int,
                                 r0: float = BASE_SHORT_RATE) -> np.ndarray:
    """
    Generate constant short-rate paths (flat r_t = r0) for testing.
    """
    return np.full((n_paths, n_steps), r0, dtype=float)

# Discount factors from short-rate path 
def discount_factors_from_short_rate_path(r_path: np.ndarray,
                                          dt_months: int = DT_MONTHS) -> np.ndarray:
    """
    Given a 1D short-rate path r_t, compute discount factors DF_t.

    We assume:
      - r_t is annualized short rate
      - dt_months is the time step size in months (usually 1)
      - DF_t = exp( - sum_{i=0}^t r_i * dt_years )
    """
    dt_years = dt_months / 12.0
    cumulative = np.cumsum(r_path * dt_years)
    df = np.exp(-cumulative)
    return df

# Hull–White / OU short-rate model 
def simulate_hull_white_paths(n_paths: int,
                              n_steps: int,
                              a: float = HW_A,
                              sigma: float = HW_SIGMA,
                              r0: float = None,
                              theta: float = None) -> np.ndarray:
    """
    Simulate short-rate paths using a simple discrete-time Hull-White / OU process:

        r_{t+1} = r_t + a (theta - r_t) dt + sigma * sqrt(dt) * eps_t

    where:
        - theta is the long-run mean short rate.
        - eps_t ~ N(0, 1)

    Parameters
    ----------
    n_paths : int
        Number of Monte Carlo paths.
    n_steps : int
        Number of time steps.
    a : float
        Mean-reversion speed.
    sigma : float
        Volatility.
    r0 : float, optional
        Initial short rate. If None, we will infer it from the short end
        of the SOFR zero curve, or fall back to BASE_SHORT_RATE.
    theta : float, optional
        Long-run mean for the short rate. If None, we will use the
        longer end of the SOFR zero curve, or fall back to BASE_SHORT_RATE.

    Returns
    -------
    np.ndarray
        Shape: (n_paths, n_steps) array of short rates.
    """
    dt_years = DT_MONTHS / 12.0

    # Build zero curve and infer r0 / theta if not provided
    zero_curve = build_zero_curve()
    tenors = sorted(zero_curve.keys())

    if r0 is None:
        # Use the shortest maturity as r0
        r0 = zero_curve[tenors[0]] if tenors else BASE_SHORT_RATE

    if theta is None:
        # Use an average of longer maturities as theta (e.g. >= 5y),
        # or the last point if no long tenors exist.
        long_tenors = [t for t in tenors if t >= 5.0]
        if long_tenors:
            long_rates = [zero_curve[t] for t in long_tenors]
            theta = float(np.mean(long_rates))
        else:
            theta = zero_curve[tenors[-1]] if tenors else BASE_SHORT_RATE

    r_paths = np.zeros((n_paths, n_steps), dtype=float)
    r_paths[:, 0] = r0

    # Pre-draw all shocks
    eps = np.random.normal(size=(n_paths, n_steps - 1))

    for t in range(n_steps - 1):
        r_t = r_paths[:, t]
        dr = a * (theta - r_t) * dt_years + sigma * np.sqrt(dt_years) * eps[:, t]
        r_paths[:, t + 1] = r_t + dr

    return r_paths
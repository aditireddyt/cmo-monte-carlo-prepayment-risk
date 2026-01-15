import re
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from config import CURVE_FILE

def _parse_tenor_to_years(tenor: str) -> float:
    """
    Convert Bloomberg tenor strings like '1W', '3M', '5Y'
    into year fractions.

    Approximation:
      - D: days / 365
      - W: weeks / 52
      - M: months / 12
      - Y: years
    """
    if not isinstance(tenor, str):
        return np.nan

    tenor = tenor.strip().upper()
    m = re.match(r"(\d+)([DWMY])", tenor)
    if not m:
        return np.nan

    num = float(m.group(1))
    unit = m.group(2)

    if unit == "D":
        return num / 365.0
    elif unit == "W":
        return num / 52.0
    elif unit == "M":
        return num / 12.0
    elif unit == "Y":
        return num
    else:
        return np.nan

def load_sofr_curve() -> pd.DataFrame:
    """
    Load the USD SOFR swap curve exported from Bloomberg
    (usd_sofr_curve.xlsx) and return a cleaned DataFrame
    with tenor in years and yield in decimal.

    Expected columns in the Excel:
      - 'Tenor'
      - 'Yield'

    Returns
    DataFrame
        Columns:
          - t_years : float (tenor in years)
          - rate    : float (annualized yield, decimal)
    """
    df = pd.read_excel(CURVE_FILE)

    if "Tenor" not in df.columns or "Yield" not in df.columns:
        raise ValueError(
            f"Expected columns 'Tenor' and 'Yield' in {CURVE_FILE}, "
            f"found {df.columns.tolist()}"
        )

    df = df[["Tenor", "Yield"]].copy()
    df = df.dropna(subset=["Tenor", "Yield"])

    # Convert tenor to years
    df["t_years"] = df["Tenor"].apply(_parse_tenor_to_years)
    df = df.dropna(subset=["t_years"])

    # Convert Yield to decimal (Bloomberg gives %)
    # If numbers look like 3.98, that's 3.98% -> 0.0398
    if df["Yield"].mean() > 1.0:
        df["rate"] = df["Yield"] / 100.0
    else:
        df["rate"] = df["Yield"].astype(float)

    df = df[["t_years", "rate"]].dropna()
    df = df.sort_values("t_years").reset_index(drop=True)

    return df

def build_zero_curve_from_sofr() -> Dict[float, float]:
    """
    Build a simple zero curve dictionary from the SOFR swap curve.

    For now, we treat the quoted swap yields as zero rates at
    those maturities

    Returns
    dict
        Mapping:
          t_years (float) -> annualized zero rate (decimal)
    """
    df = load_sofr_curve()
    zero_curve = dict(zip(df["t_years"].values, df["rate"].values))
    return zero_curve

def interpolate_zero_rate(t: float, zero_curve: Dict[float, float]) -> float:
    """
    Linearly interpolate the zero rate at time t (in years)
    from a zero_curve dictionary.

    If t is outside the curve range, use the closest endpoint.
    """
    if t < 0:
        t = 0.0

    knots = np.array(sorted(zero_curve.keys()))
    rates = np.array([zero_curve[k] for k in knots])

    if t <= knots[0]:
        return rates[0]
    if t >= knots[-1]:
        return rates[-1]

    # Find where t lies
    idx = np.searchsorted(knots, t) 
    t1 = knots[idx - 1]
    t2 = knots[idx]
    r1 = rates[idx - 1]
    r2 = rates[idx]

    # Linear interpolation
    w = (t - t1) / (t2 - t1)
    return r1 * (1.0 - w) + r2 * w
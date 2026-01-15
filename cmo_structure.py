from dataclasses import dataclass
from typing import Optional, Dict, List
import pandas as pd

@dataclass
class Tranche:
    """
    Tranche representation for the CMO structure.
    Attributes
    name : str
        Short name, e.g. "PA", "PI", "PL", "W", "WA", "WB", "WC".
    initial_balance : float
        Starting principal balance.
    coupon : float
        Annual coupon as decimal (e.g. 0.05 for 5%).
    tranche_type : str
        E.g. "PAC", "Support", "Z". For now we can label
        PA/PI/PL as "PAC", others as "Support" if we like.
    priority : int
        Payment priority in the waterfall. Lower = more senior.
    """
    name: str
    initial_balance: float
    coupon: float
    tranche_type: str
    priority: int = 1

def _build_tranche_from_bond_sheet(short_name: str,
                                   bond_df: pd.DataFrame,
                                   tranche_type: str,
                                   priority: int) -> Tranche:
    """
    Helper to construct a Tranche from a bond sheet DataFrame.
    Expects columns:
        ['Date', 'Balance', 'Principal', 'Interest', 'Loss',
         'Shortfall', 'Coupon', 'Interest Fraction']
    """
    first_row = bond_df.iloc[0]

    initial_balance = float(first_row["Balance"])
    coupon_raw = float(first_row["Coupon"])

    # Coupon can be in percent or decimal
    if coupon_raw > 1.0:
        coupon = coupon_raw / 100.0
    else:
        coupon = coupon_raw

    return Tranche(
        name=short_name,
        initial_balance=initial_balance,
        coupon=coupon,
        tranche_type=tranche_type,
        priority=priority,
    )

def build_simple_tranche_from_bond_pa(bond_pa_df: pd.DataFrame) -> Tranche:
    """
    Backwards-compatible helper: build a single PA tranche from Bond PA sheet.
    """
    return _build_tranche_from_bond_sheet(
        short_name="PA",
        bond_df=bond_pa_df,
        tranche_type="PAC",
        priority=1,
    )

def build_tranches_from_deal(deal_data: Dict[str, pd.DataFrame]) -> List[Tranche]:
    """
    Build a list of Tranche objects from all available bond sheets in deal_data.

    Parameters
    deal_data : dict
        Output of load_deal_data(), with keys like:
          - 'bond_PA', 'bond_PI', 'bond_PL', 'bond_W',
            'bond_WA', 'bond_WB', 'bond_WC'

    Returns
    list of Tranche
        Tranches with priorities assigned in a simple sequential structure.
    """
    bond_config = [
        # short_name,  key_in_deal_data, tranche_type, priority
        ("PA", "bond_PA", "PAC",     1),
        ("PI", "bond_PI", "PAC",     2),
        ("PL", "bond_PL", "PAC",     3),
        ("W",  "bond_W",  "Support", 4),
        ("WA", "bond_WA", "Support", 5),
        ("WB", "bond_WB", "Support", 6),
        ("WC", "bond_WC", "Support", 7),
    ]

    tranches: List[Tranche] = []

    for short_name, key, ttype, prio in bond_config:
        if key in deal_data:
            bond_df = deal_data[key]
            tranche = _build_tranche_from_bond_sheet(
                short_name=short_name,
                bond_df=bond_df,
                tranche_type=ttype,
                priority=prio,
            )
            tranches.append(tranche)

    # Sort by priority just to be safe
    tranches.sort(key=lambda x: x.priority)

    return tranches
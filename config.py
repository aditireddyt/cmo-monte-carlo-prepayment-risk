DEAL_FILE = "project_data.xlsx"

# curve file from Bloomberg
CURVE_FILE = "usd_sofr_curve.xlsx"

# Simulation settings
N_PATHS = 5              # you can increase later
DT_MONTHS = 1            # monthly steps
HORIZON_YEARS = 30       # using 30 years now

# Base flat short rate for simple model 
BASE_SHORT_RATE = 0.04   # 4% annualized

# Hull–White parameters (placeholders for now)
HW_A = 0.1               # mean reversion
HW_SIGMA = 0.01          # volatility

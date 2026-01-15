# Monte Carlo CMO Prepayment & Interest Rate Risk Model

This project implements a Monte Carlo framework to analyze tranche-level cash flows, pricing, and risk metrics for agency CMOs.

## Overview
- Simulates interest-rate paths using a calibrated Hull–White model
- Models mortgage prepayments using a CPR regression
- Evaluates WAL, duration, convexity, and tail risk across tranches

## Key Findings
- Prepayment risk dominates interest-rate risk for support tranches
- Low-CPR scenarios extend support-tranche WAL by ~20% relative to PAC tranches

## Tools & Methods
- Python
- Monte Carlo simulation
- Hull–White interest-rate model
- CPR-based prepayment modeling

## Files
- `simulate.py`: Monte Carlo engine
- `waterfall.py`: Tranche cash-flow waterfall
- `rates.py`: Interest-rate modeling
- `data_prepay.py`: Prepayment regression

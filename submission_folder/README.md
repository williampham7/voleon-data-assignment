PORTFOLIO RISK REPORT GENERATOR

Daily risk report for factor neutral global equities portfolio. Reviews 
positions and identifies unintended exposures before market open.

REQUIREMENTS

python
pandas
numpy

USAGE

pPaste this into the terminal when you are in the directory containing the report 
generator file and spreadsheets:

python3 portfolio_report.py positions.csv fx.csv

INPUT FILES

positions.csv - Portfolio positions
fx.csv - FX rates

OUTPUT

Report prints to console and saves to portfolio_risk_report.txt

WHAT THE REPORT SHOWS

  Unintended Exposures - Warnings for beta, concentration, and liquidity risks
  Portfolio Summary - Total exposures, P&L, position counts, beta analysis
  Market Shock Analysis - P&L impact from 2% market move
  Concentration Analysis - Top positions, sector/country/currency breakdowns
  Liquidity Analysis - Days to unwind positions (assumes 10% of daily volume)

KEY WARNINGS

The report flags:
  Portfolio beta > ±0.1 (should be ~0 for factor neutral)
  Net exposure > ±10% of GMV
  Sector/country/currency concentration > 30% of GMV
  Single name > 5% of GMV
  Positions requiring >10 days to unwind
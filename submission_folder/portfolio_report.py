# python3.12 portfolio_report.py positions.csv fx.csv

"""
Factor Neutral Global Equities Portfolio risk report generator
Generates a daily risk report before market open

Inputs:
    positions.csv - portfolio positions data
    fx.csv - foreign exchange rates to USD

Output:
    plaintext report printed to console and saved to portfolio_risk_report.txt
"""

import pandas as pd
import numpy as np
import sys
from datetime import datetime

def load_data(positions_file, fx_file):
    """Load portfolio positions and FX rates"""
    portfolio = pd.read_csv(positions_file)
    fx_rates = pd.read_csv(fx_file)
    
    # removes unnamed columns if any
    fx_rates = fx_rates[[col for col in fx_rates.columns if 'Unnamed' not in col]]
    
    return portfolio, fx_rates


def clean_data(portfolio, fx_rates):    
    # merge fx rates into portfolio df
    portfolio = portfolio.merge(fx_rates, on='currency', how='left')
    
    # there are some missing currency codes, fill based on country
    currency_map = {
        'AUS': 'AUD',
        'USA': 'USD',
        'GBR': 'GBP',
        'JPN': 'JPY',
        'CHE': 'CHF',
        'CAN': 'CAD',
        'FRA': 'EUR',
        'GER': 'EUR',
        'ITA': 'EUR',
        'ESP': 'EUR',
        'NLD': 'EUR',
        'BEL': 'EUR',
        'CHN': 'CNY',
        'HKG': 'HKD',
        'BRA': 'BRL'
    }
    
    for idx, row in portfolio[portfolio['currency'].isna()].iterrows():
        if row['country'] in currency_map:
            portfolio.at[idx, 'currency'] = currency_map[row['country']]
    
    # merge FX rates for filled currencies
    portfolio = portfolio.drop(columns=['to_USD'], errors='ignore')
    portfolio = portfolio.merge(fx_rates, on='currency', how='left')
    
    # calculate position values
    portfolio['position_value_local'] = portfolio['posn_shares'] * portfolio['market_price_local']
    portfolio['cost_value_local'] = portfolio['posn_shares'] * portfolio['cost_basis_local']
    portfolio['unrealized_pnl_local'] = portfolio['position_value_local'] - portfolio['cost_value_local']
    
    # convert to USD
    portfolio['position_value_usd'] = portfolio['position_value_local'] * portfolio['to_USD']
    portfolio['cost_value_usd'] = portfolio['cost_value_local'] * portfolio['to_USD']
    portfolio['unrealized_pnl_usd'] = portfolio['unrealized_pnl_local'] * portfolio['to_USD']
    
    # Calculate portfolio weights
    total_gmv = portfolio['position_value_usd'].abs().sum()
    portfolio['position_weight'] = portfolio['position_value_usd'] / total_gmv
    portfolio['dollar_weight'] = portfolio['position_value_usd'].abs() / total_gmv
    
    # Calculate days to unwind (we assume max 10% daily value can be traded per day)
    portfolio['days_to_unwind'] = (portfolio['posn_shares'].abs() / 
                                    (portfolio['avg_daily_volume'] * 0.10))

    return portfolio, total_gmv


def generate_report_header():
    """Generate report header with timestamp"""
    now = datetime.now()
    header = f"""
FACTOR NEUTRAL GLOBAL EQUITIES PORTFOLIO - DAILY RISK REPORT
{'='*80}
Report Generated: {now.strftime('%Y-%m-%d %H:%M:%S')}

"""
    return header


def portfolio_summary_section(portfolio, total_gmv):
    """Generate portfolio summary statistics"""
    
    long_value = portfolio[portfolio['position_value_usd'] > 0]['position_value_usd'].sum()
    short_value = portfolio[portfolio['position_value_usd'] < 0]['position_value_usd'].sum()
    net_value = portfolio['position_value_usd'].sum()
    total_pnl = portfolio['unrealized_pnl_usd'].sum()

    market_shock = 0.02
    portfolio['shock_pnl'] = (portfolio['beta'] * market_shock * portfolio['position_value_usd'])
    simulated_loss = portfolio['shock_pnl'].sum()
    simulated_loss_pct = portfolio['shock_pnl'].sum() / total_gmv

    weighted_beta = (portfolio['beta'] * portfolio['position_value_usd']).sum() / total_gmv
    long_beta = (portfolio[portfolio['side'] == 'LONG']['beta'] * 
                 portfolio[portfolio['side'] == 'LONG']['position_value_usd']).sum() / total_gmv
    short_beta = (portfolio[portfolio['side'] == 'SHORT']['beta'] * 
                  portfolio[portfolio['side'] == 'SHORT']['position_value_usd']).sum() / total_gmv

    
    section = f"""
PORTFOLIO SUMMARY
{'-'*80}
Total Positions:                       {len(portfolio)}
Long Positions:                        {len(portfolio[portfolio['side'] == 'LONG'])}
Short Positions:                       {len(portfolio[portfolio['side'] == 'SHORT'])}
Long/Short Ratio:                      {len(portfolio[portfolio['side'] == 'LONG']) / max(len(portfolio[portfolio['side'] == 'SHORT']), 1):.2f}

Total Gross Market Value (GMV):        ${total_gmv:,.2f}
Total Long Exposure:                   ${long_value:,.2f}
Total Short Exposure:                  ${short_value:,.2f}
Net Market Exposure:                   ${net_value:,.2f}
Net Exposure / GMV:                    {(net_value/total_gmv)*100:.2f}%

Total Unrealized P&L:                  ${total_pnl:,.2f}
P&L / GMV:                             {(total_pnl/total_gmv)*100:.2f}%

FACTOR EXPOSURES
{'-'*80}
Portfolio Beta:                        {weighted_beta:.4f}
Long Book Beta:                        {long_beta:.4f}
Short Book Beta:                       {short_beta:.4f}

MARKET SHOCK SENSITIVITY ANALYSIS
{'-'*80}
MARKET SHOCK SCENARIO (2% Market Move):
Expected P&L Impact:                   ${simulated_loss:,.2f}
Impact as % of GMV:                    {simulated_loss_pct:.4%}
"""
    return section


def concentration_analysis_section(portfolio):
    """Analyze concentrations by various dimensions"""
    gmv = portfolio['position_value_usd'].abs().sum()

    # Top positions by absolute value
    top_positions = portfolio.nlargest(10, 'dollar_weight')[
        ['ticker', 'name', 'country', 'sector', 'position_value_usd', 'dollar_weight', 'side']
    ]

    # Sector concentrations
    sector_exposure = portfolio.groupby('sector').agg(
        Net_Exposure_USD=('position_value_usd', 'sum'),
        Gross_Exposure_USD=('position_value_usd', lambda x: x.abs().sum()),
        Unrealized_PnL=('unrealized_pnl_usd', 'sum')
    )
    sector_exposure['Pct_of_GMV'] = sector_exposure['Gross_Exposure_USD'] / gmv
    sector_exposure = sector_exposure.round(2).sort_values('Pct_of_GMV', ascending=False, key=np.abs)
    # Reset index so the label (sector) prints as a normal column header in to_string
    sector_exposure_print = sector_exposure.reset_index()

    # Country concentrations
    country_exposure = portfolio.groupby('country').agg(
        Net_Exposure_USD=('position_value_usd', 'sum'),
        Gross_Exposure_USD=('position_value_usd', lambda x: x.abs().sum()),
        Unrealized_PnL=('unrealized_pnl_usd', 'sum')
    ).round(2)
    country_exposure['Pct_of_GMV'] = country_exposure['Gross_Exposure_USD'] / gmv
    country_exposure = country_exposure.sort_values('Pct_of_GMV', ascending=False, key=np.abs)
    # Reset index so the label (country) prints inline with headers
    country_exposure_print = country_exposure.reset_index()
    
    # Currency exposures
    currency_exposure = portfolio.groupby('currency').agg(
        Net_Exposure_USD=('position_value_usd', 'sum'),
        Gross_Exposure_USD=('position_value_usd', lambda x: x.abs().sum()),
        Unrealized_PnL=('unrealized_pnl_usd', 'sum')
    ).round(2)
    currency_exposure['Pct_of_GMV'] = currency_exposure['Gross_Exposure_USD'] / gmv
    currency_exposure = currency_exposure.sort_values('Pct_of_GMV', ascending=False, key=np.abs)
    currency_exposure_print = currency_exposure.reset_index()
    
    section = f"""
CONCENTRATION ANALYSIS
{'-'*80}

SECTOR EXPOSURE:
{sector_exposure_print.to_string(index=False)}

COUNTRY EXPOSURE:
{country_exposure_print.to_string(index=False)}

CURRENCY EXPOSURE:
{currency_exposure_print.to_string(index=False)}

TOP 10 POSITIONS BY SIZE:
{top_positions.to_string(index=False)}

"""
    return section


def liquidity_analysis_section(portfolio):
    """Analyze portfolio liquidity and potential unwind risk"""
    
    # Positions that would take >5 days to unwind
    illiquid_positions = portfolio[portfolio['days_to_unwind'] > 5.0].sort_values(
        'days_to_unwind', ascending=False
    )[['ticker', 'name', 'country', 'posn_shares', 'avg_daily_volume', 
       'days_to_unwind', 'position_value_usd']]
    
    # Summary stats
    avg_days = portfolio['days_to_unwind'].mean()
    median_days = portfolio['days_to_unwind'].median()
    max_days = portfolio['days_to_unwind'].max()
    
    section = f"""
LIQUIDITY ANALYSIS
{'-'*80}
Assuming that 10% of average daily volume can be traded per day:

Median Days to Unwind:                 {median_days:.2f} days
Number of Illiquid Positions (>5d):    {len(illiquid_positions)}

"""
    
    if len(illiquid_positions) > 0:
        section += f"""
MOST ILLIQUID POSITIONS:
{illiquid_positions.head(10).to_string(index=False)}
"""
    
    return section

def unintended_exposures_section(portfolio, total_gmv):
    """Flag potential unintended exposures"""
    
    warnings = []
    
    # Check beta neutrality
    weighted_beta = (portfolio['beta'] * portfolio['position_value_usd']).sum() / total_gmv
    if abs(weighted_beta) > 0.1:
        warnings.append(f"⚠️  CRITICAL: Portfolio beta ({weighted_beta:.4f}) exceeds neutral threshold (±0.1)")
    
    # Check net exposure
    net_exposure = portfolio['position_value_usd'].sum() / total_gmv
    if abs(net_exposure) > 0.1:
        warnings.append(f"⚠️  Net market exposure ({net_exposure:.2%}) exceeds threshold (±10%)")
    
    # Check sector concentrations
    sector_exposure = portfolio.groupby('sector')['position_value_usd'].sum().abs() / total_gmv
    max_sector = sector_exposure.max()
    if max_sector > 0.3:
        warnings.append(f"Sector concentration ({max_sector:.2%}) in {sector_exposure.idxmax()}")
    
    # Check country concentrations
    country_exposure = portfolio.groupby('country')['position_value_usd'].sum().abs() / total_gmv
    max_country = country_exposure.max()
    if max_country > 0.3:
        warnings.append(f"Country concentration ({max_country:.2%}) in {country_exposure.idxmax()}")

    # Check currency concentrations
    currency_exposure = portfolio.groupby('currency')['position_value_usd'].agg(lambda x: x.abs().sum()) / total_gmv
    max_currency = currency_exposure.max()
    if max_currency > 0.3:
        warnings.append(f"High Currency concentration ({max_currency:.2%}) in {currency_exposure.idxmax()}")

    # Check single name concentrations
    max_single = portfolio['dollar_weight'].max()
    if max_single > 0.05:
        ticker = portfolio.loc[portfolio['dollar_weight'].idxmax(), 'ticker']
        warnings.append(f"Single name concentration ({max_single:.2%}) in {ticker}")
    
    # Check for illiquid large positions
    large_illiquid = portfolio[(portfolio['days_to_unwind'] > 10)]
    if len(large_illiquid) > 0:
        warnings.append(f"{len(large_illiquid)} large positions require >10 days to unwind")
    

    section = f"""
UNINTENDED EXPOSURE WARNINGS
{'-'*80}
"""
    
    if warnings:
        for warning in warnings:
            section += f"{warning}\n"
    else:
        section += "✓ No significant unintended exposures detected\n"
    
    section += "\n"
    
    return section


def generate_footer():
    """Generate report footer"""
    footer = f"""
{'='*80}
END OF REPORT
{'='*80}
"""
    return footer


def main():
    """Main function to generate the report"""
    
    # Check command line arguments
    if len(sys.argv) != 3:
        print("Usage: python portfolio_report.py <positions.csv> <fx.csv>")
        sys.exit(1)
    
    positions_file = sys.argv[1]
    fx_file = sys.argv[2]
    
    try:
        portfolio, fx_rates = load_data(positions_file, fx_file)
        
        portfolio, total_gmv = clean_data(portfolio, fx_rates)
        
        report = ""
        report += generate_report_header()
        report += unintended_exposures_section(portfolio, total_gmv)
        report += portfolio_summary_section(portfolio, total_gmv)
        report += concentration_analysis_section(portfolio)
        report += liquidity_analysis_section(portfolio)
        report += generate_footer()
        
        print(report)
        
        # Save to .txt file
        output_file = "portfolio_risk_report.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n✓ Report saved to {output_file}")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
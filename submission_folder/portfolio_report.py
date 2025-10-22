# python portfolio_report.py positions.csv fx.csv

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
    
    # Re-merge FX rates for filled currencies
    portfolio = portfolio.drop(columns=['to_USD'], errors='ignore')
    portfolio = portfolio.merge(fx_rates, on='currency', how='left')
    
    # Calculate position values
    portfolio['position_value_local'] = portfolio['posn_shares'] * portfolio['market_price_local']
    portfolio['cost_value_local'] = portfolio['posn_shares'] * portfolio['cost_basis_local']
    portfolio['unrealized_pnl_local'] = portfolio['position_value_local'] - portfolio['cost_value_local']
    
    # Convert to USD
    portfolio['position_value_usd'] = portfolio['position_value_local'] * portfolio['to_USD']
    portfolio['cost_value_usd'] = portfolio['cost_value_local'] * portfolio['to_USD']
    portfolio['unrealized_pnl_usd'] = portfolio['unrealized_pnl_local'] * portfolio['to_USD']
    
    # Calculate portfolio weights
    total_gmv = portfolio['position_value_usd'].abs().sum()
    portfolio['position_weight'] = portfolio['position_value_usd'] / total_gmv
    portfolio['dollar_weight'] = portfolio['position_value_usd'].abs() / total_gmv
    
    # Calculate days to unwind (assuming we can trade 10% of daily volume)
    portfolio['days_to_unwind'] = (portfolio['posn_shares'].abs() / 
                                    (portfolio['avg_daily_volume'] * 0.10))

    return portfolio, total_gmv


def generate_report_header():
    """Generate report header with timestamp"""
    now = datetime.now()
    header = f"""
{'='*80}
FACTOR NEUTRAL GLOBAL EQUITIES PORTFOLIO - DAILY RISK REPORT
{'='*80}
Report Generated: {now.strftime('%Y-%m-%d %H:%M:%S')}
Report Date: {now.strftime('%A, %B %d, %Y')}
{'='*80}

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

    
    section = f"""
PORTFOLIO SUMMARY
{'-'*80}
Number of Long Positions:              {len(portfolio[portfolio['side'] == 'LONG'])}
Number of Short Positions:             {len(portfolio[portfolio['side'] == 'SHORT'])}
Total Positions:                       {len(portfolio)}

Total Gross Market Value (GMV):        ${total_gmv:,.2f}
Total Long Exposure:                   ${long_value:,.2f}
Total Short Exposure:                  ${short_value:,.2f}
Net Market Exposure:                   ${net_value:,.2f}
Net Exposure / GMV:                    {(net_value/total_gmv)*100:.2f}%

Total Unrealized P&L:                  ${total_pnl:,.2f}
P&L / GMV:                             {(total_pnl/total_gmv)*100:.2f}%

MARKET SHOCK SENSITIVITY
{'-'*80}
Simulated P&L under 2% market shock
Net Gain/Loss: {simulated_loss:.6%}")
Net Gain/Loss as % of GMV: {simulated_loss_pct:.6%}")
"""
    return section


def factor_exposures_section(portfolio, total_gmv):
    """Analyze factor exposures - key for factor neutral strategy"""
    
    # Beta exposure
    weighted_beta = (portfolio['beta'] * portfolio['position_value_usd']).sum() / total_gmv
    long_beta = (portfolio[portfolio['side'] == 'LONG']['beta'] * 
                 portfolio[portfolio['side'] == 'LONG']['position_value_usd']).sum() / total_gmv
    short_beta = (portfolio[portfolio['side'] == 'SHORT']['beta'] * 
                  portfolio[portfolio['side'] == 'SHORT']['position_value_usd']).sum() / total_gmv
    
    section = f"""
FACTOR EXPOSURES
{'-'*80}
Portfolio Beta:                      {weighted_beta:.4f}
Long Book Beta:                      {long_beta:.4f}
Short Book Beta:                     {short_beta:.4f}
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

    # Country concentrations
    country_exposure = portfolio.groupby('country').agg(
        Net_Exposure_USD=('position_value_usd', 'sum'),
        Gross_Exposure_USD=('position_value_usd', lambda x: x.abs().sum()),
        Unrealized_PnL=('unrealized_pnl_usd', 'sum')
    ).round(2)
    country_exposure['Pct_of_GMV'] = country_exposure['Gross_Exposure_USD'] / gmv
    country_exposure = country_exposure.sort_values('Pct_of_GMV', ascending=False, key=np.abs)
    
    # Currency exposures
    currency_exposure = portfolio.groupby('currency').agg(
        Net_Exposure_USD=('position_value_usd', 'sum'),
        Gross_Exposure_USD=('position_value_usd', lambda x: x.abs().sum()),
        Unrealized_PnL=('unrealized_pnl_usd', 'sum')
    ).round(2)
    currency_exposure['Pct_of_GMV'] = currency_exposure['Gross_Exposure_USD'] / gmv
    currency_exposure = currency_exposure.sort_values('Pct_of_GMV', ascending=False, key=np.abs)
    
    section = f"""
CONCENTRATION ANALYSIS
{'-'*80}

TOP 10 POSITIONS BY SIZE:
{top_positions.to_string(index=False)}

SECTOR EXPOSURE:
{sector_exposure.to_string()}

COUNTRY EXPOSURE:
{country_exposure.to_string()}

CURRENCY EXPOSURE:
{currency_exposure.to_string()}

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
Average Days to Unwind:                {avg_days:.2f} days
Median Days to Unwind:                 {median_days:.2f} days
Maximum Days to Unwind:                {max_days:.2f} days

Number of Illiquid Positions (>5d):    {len(illiquid_positions)}

"""
    
    if len(illiquid_positions) > 0:
        section += f"""⚠️  ILLIQUID POSITIONS REQUIRING ATTENTION:
                {illiquid_positions.head(10).to_string(index=False)}

                """
    
    return section


def risk_analysis_section(portfolio, total_gmv):
    """Perform risk analysis including market shock scenarios"""
    
    # Simulate 2% market shock
    market_shock = 0.02
    portfolio['shock_pnl'] = portfolio['beta'] * market_shock * portfolio['position_value_usd']
    
    simulated_pnl = portfolio['shock_pnl'].sum()
    simulated_pnl_pct = (simulated_pnl / total_gmv) * 100
    
    # Find positions most at risk
    worst_in_shock = portfolio.nsmallest(5, 'shock_pnl')[
        ['ticker', 'name', 'beta', 'position_value_usd', 'shock_pnl', 'side']
    ]
    
    # High beta positions
    high_beta = portfolio[portfolio['beta'].abs() > 2.0].sort_values('beta', ascending=False)[
        ['ticker', 'name', 'beta', 'position_value_usd', 'dollar_weight', 'side']
    ]
    
    section = f"""
RISK ANALYSIS
{'-'*80}

MARKET SHOCK SCENARIO (2% Market Move):
Expected P&L Impact:                   ${simulated_pnl:,.2f}
Impact as % of GMV:                    {simulated_pnl_pct:.4f}%

⚠️  Note: For a truly factor neutral portfolio, this should be ~0%
    Current exposure suggests portfolio will {'GAIN' if simulated_pnl > 0 else 'LOSE'} in market rally

POSITIONS MOST AT RISK IN MARKET SHOCK:
{worst_in_shock.to_string(index=False)}

HIGH BETA POSITIONS (|Beta| > 2.0):
Number of High Beta Positions:         {len(high_beta)}

"""
    
    if len(high_beta) > 0:
        section += f"{high_beta.head(10).to_string(index=False)}"
    
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
        warnings.append(f"⚠️  Sector concentration ({max_sector:.2%}) in {sector_exposure.idxmax()}")
    
    # Check country concentrations
    country_exposure = portfolio.groupby('country')['position_value_usd'].sum().abs() / total_gmv
    max_country = country_exposure.max()
    if max_country > 0.4:
        warnings.append(f"⚠️  Country concentration ({max_country:.2%}) in {country_exposure.idxmax()}")
    
    # Check single name concentrations
    max_single = portfolio['dollar_weight'].max()
    if max_single > 0.05:
        ticker = portfolio.loc[portfolio['dollar_weight'].idxmax(), 'ticker']
        warnings.append(f"⚠️  Single name concentration ({max_single:.2%}) in {ticker}")
    
    # Check for illiquid large positions
    large_illiquid = portfolio[(portfolio['days_to_unwind'] > 10) & 
                               (portfolio['dollar_weight'] > 0.02)]
    if len(large_illiquid) > 0:
        warnings.append(f"⚠️  {len(large_illiquid)} large positions require >10 days to unwind")
    
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

IMPORTANT NOTES:
- This report should be reviewed daily before market open
- Factor neutral portfolios should maintain beta ~0 and net exposure ~0
- Review all flagged warnings and consider rebalancing if necessary
- Contact Risk Management for questions or concerns

Report Generation: Automated via portfolio_report.py
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
        # Load and process data
        print("Loading data...")
        portfolio, fx_rates = load_data(positions_file, fx_file)
        
        print("Processing portfolio...")
        portfolio, total_gmv = clean_data(portfolio, fx_rates)
        
        # Generate report sections
        print("Generating report...")
        report = ""
        report += generate_report_header()
        report += portfolio_summary_section(portfolio, total_gmv)
        report += factor_exposures_section(portfolio, total_gmv)
        report += concentration_analysis_section(portfolio)
        report += liquidity_analysis_section(portfolio)
        report += risk_analysis_section(portfolio, total_gmv)
        report += unintended_exposures_section(portfolio, total_gmv)
        report += generate_footer()
        
        # Print to console
        print(report)
        
        # Save to file
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
import pandas as pd
import numpy as np
from scipy.stats import norm


def create_futures_data(commodity='CORN', start_date='2010-01-01', end_date='2025-12-31'):
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    dt = 1/252  # Daily time step (approximately 252 trading days per year)
    
    # Contract months for CBOT
    months = ['H', 'K', 'N', 'U', 'Z']  # Mar, May, Jul, Sep, Dec
    months_dict = {
        'H': 3, 'K': 5, 'N': 7, 'U': 9, 'Z': 12
    }
    
    # Commodity-specific parameters
    params = {
        'CORN': {
            'S0': 500,  # Initial spot price
            'kappa': 0.5,  # Mean reversion speed
            'theta': 550,  # Long-term mean price
            'sigma': 0.3,  # Volatility
            'seasonal_amp': 50,  # Seasonal amplitude
            'storage_cost': 0.15,  # Annual storage cost as fraction of spot
            'convenience_yield': 0.1,  # Annual convenience yield
        },
        'WHEAT': {
            'S0': 600,
            'kappa': 0.4,
            'theta': 650,
            'sigma': 0.25,
            'seasonal_amp': 60,
            'storage_cost': 0.12,
            'convenience_yield': 0.08,
        },
        'SOYBEAN': {
            'S0': 1200,
            'kappa': 0.3,
            'theta': 1300,
            'sigma': 0.2,
            'seasonal_amp': 100,
            'storage_cost': 0.1,
            'convenience_yield': 0.06,
        }
    }
    
    def seasonal_factor(t, commodity_params):
        """Calculate seasonal component"""
        # Peak in summer for agricultural commodities
        return commodity_params['seasonal_amp'] * np.sin(2 * np.pi * (t + 0.3))
    
    def generate_spot_prices(dates, params):
        """Generate spot prices using Ornstein-Uhlenbeck process with seasonality"""
        n_steps = len(dates)
        prices = np.zeros(n_steps)
        prices[0] = params['S0']
        
        for t in range(1, n_steps):
            # Seasonal adjustment to mean
            theta_t = params['theta'] + seasonal_factor(dates[t].month/12, params)
            
            # Ornstein-Uhlenbeck process
            dW = np.random.normal(0, np.sqrt(dt))
            dS = params['kappa'] * (theta_t - prices[t-1]) * dt + \
                 params['sigma'] * prices[t-1] * dW
            prices[t] = prices[t-1] + dS
        
        return prices
    
    def calculate_futures_price(spot_price, time_to_maturity, params):
        """Calculate futures price using cost-of-carry model with seasonality"""
        # Adjust convenience yield for seasonality
        seasonal_adj = seasonal_factor(time_to_maturity, params)
        adj_convenience_yield = params['convenience_yield'] + 0.05 * np.sin(2 * np.pi * time_to_maturity)
        
        # Cost of carry model with convenience yield
        basis = (params['storage_cost'] - adj_convenience_yield) * time_to_maturity
        futures_price = spot_price * np.exp(basis) + seasonal_adj
        
        return futures_price
    
    # Generate spot prices
    spot_prices = generate_spot_prices(dates, params[commodity])
    
    # Create empty dataset
    dataset = pd.DataFrame(index=dates)
    dataset.index.name = 'Date'
    
    # Generate futures contracts
    for year in range(2010, 2026):
        for month in months:
            contract_name = f'{month}{str(year)[2:]}'
            
            # Calculate contract expiry
            expiry_month = months_dict[month]
            expiry_date = pd.Timestamp(f'{year}-{expiry_month:02d}-15')
            
            # Start trading 2 years before expiry (to ensure enough contracts are available)
            start_date_contract = expiry_date - pd.DateOffset(months=24)
            
            # Generate futures prices
            mask = (dates >= start_date_contract) & (dates <= expiry_date)
            contract_dates = dates[mask]
            
            if len(contract_dates) == 0:
                continue
            
            # Calculate time to maturity for each date
            time_to_maturity = [(expiry_date - date).days/365 for date in contract_dates]
            
            # Generate futures prices
            spot_idx = [dates.get_loc(date) for date in contract_dates]
            futures_prices = [calculate_futures_price(spot_prices[idx], ttm, params[commodity])
                            for idx, ttm in zip(spot_idx, time_to_maturity)]
            
            dataset[contract_name] = pd.Series(futures_prices, index=contract_dates)
    
    return dataset

def get_front_contracts(date, df, n_contracts=10):
    """Get the n front contracts for a specific date."""
    if isinstance(date, str):
        date = pd.Timestamp(date)
    
    # Get all contracts that have data for this date
    available_contracts = df.loc[date].dropna()
    
    if len(available_contracts) == 0:
        return []
    
    # Sort contracts by expiry date
    contract_expiries = {}
    for contract in available_contracts.index:
        month = contract[0]
        year = int(contract[1:])
        month_num = {'H': 3, 'K': 5, 'N': 7, 'U': 9, 'Z': 12}[month]
        full_year = 2000 + year if year < 100 else year
        expiry = pd.Timestamp(f'{full_year}-{month_num:02d}-15')
        contract_expiries[contract] = expiry
    
    # Sort contracts by expiry
    sorted_contracts = sorted(contract_expiries.items(), key=lambda x: x[1])
    
    # Get the front n contracts, ensure we get exactly n_contracts if available
    front_contracts = [contract for contract, _ in sorted_contracts[:n_contracts]]
    
    # Debug print to see what's happening
    print(f"Date: {date}, Available contracts: {len(sorted_contracts)}, Front contracts: {len(front_contracts)}")
    print(f"Front contracts: {front_contracts}")
    
    return front_contracts

def get_forward_curves(df, dates):
    """Get forward curves for a list of dates."""
    if isinstance(dates, (pd.Timestamp, str)):
        dates = [pd.Timestamp(dates)]
    elif isinstance(dates, tuple):
        start_date, end_date = dates
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    curves = {}
    for date in dates:
        if date in df.index:
            front_contracts = get_front_contracts(df=df, date=date)
            if front_contracts:  # Check if we have any contracts
                curves[date] = pd.Series(df.loc[date, front_contracts])
    return curves

# Add this for debugging
if __name__ == "__main__":
    # Test the new functionality
    corn_futures = create_futures_data(commodity='CORN')
    
    # Test single date forward curve
    print("\nSingle date forward curve (2022-01-15):")
    front_contracts = get_front_contracts(date='2022-01-15', df=corn_futures)
    print(front_contracts)
    
    # Test multiple date forward curves
    print("\nForward curves from 2022-01-15 to 2022-01-31:")
    forward_curves = get_forward_curves(df=corn_futures, dates=pd.date_range('2022-01-15', '2022-01-31'))
    print(forward_curves)

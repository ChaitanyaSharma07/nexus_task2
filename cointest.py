import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from itertools import combinations

nftybtickers = [
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", 
    "AXISBANK.NS", "INDUSINDBK.NS", "AUBANK.NS", "BANDHANBNK.NS", 
    "FEDERALBNK.NS", "IDFCFIRSTB.NS", "PNB.NS", "BANKBARODA.NS"
]

start = "2013-01-01"
middle = "2021-01-01"
end = "2025-12-31"

def load_data(tickers, start, end):
    print("downloading ticker data")
    

    data = yf.download(tickers, start=start, end=end, auto_adjust=False)
    
    # Select only the Adjusted Close prices
    adj_close_data = data['Adj Close']
    
    adj_close_data = adj_close_data.dropna(axis=1, how='all')
    
    return adj_close_data

all_data = load_data(nftybtickers, start, end)  

train_data = all_data.loc["2013-01-01":"2020-12-31"]

test_data = all_data.loc["2021-01-01":"2025-12-31"]

# 2. Generate all unique combinations (The "Secret" from the video)
pairs = list(combinations(nftybtickers, 2))
pairs_df = pd.DataFrame(pairs, columns=['Stock1', 'Stock2'])

# 3. Apply Correlation Filter (to save time)
def get_corr(row):
    return train_data[row['Stock1']].corr(train_data[row['Stock2']])

pairs_df['Correlation'] = pairs_df.apply(get_corr, axis=1)
# Only keep pairs with > 0.90 correlation for the cointegration test
filtered_pairs = pairs_df[pairs_df['Correlation'] > 0.90].copy()

# 4. Define the Cointegration Test (Regression + ADF)
def check_cointegration(row):
    S1 = train_data[row['Stock1']]
    S2 = train_data[row['Stock2']]
    
    # Run OLS Regression
    X = sm.add_constant(S1)
    model = sm.OLS(S2, X).fit()
    
    # Test residuals for stationarity (The Engle-Granger Step)
    result = adfuller(model.resid)
    p_value = result[1]
    t_stat = result[0]
    
    return pd.Series([t_stat, p_value], index=['T-Stat', 'P-Value'])

# 5. Run the test on filtered pairs
results = filtered_pairs.apply(check_cointegration, axis=1)
final_report = pd.concat([filtered_pairs, results], axis=1)

# 6. Filter for statistically significant pairs (P-Value < 0.05)
significant_pairs = final_report[final_report['P-Value'] < 0.05].sort_values(by='P-Value')

print("Significant Cointegrated Pairs found:")
print(significant_pairs)

if not significant_pairs.empty:
    best_pair = significant_pairs.iloc[0]
    stock1 = best_pair['Stock1']
    stock2 = best_pair['Stock2']
    print(f"\nSelected Best Pair: {stock1} vs {stock2} (P-Value: {best_pair['P-Value']:.5f})")
else:
    print("\nNo significant pairs found! Falling back to HDFCBANK.NS vs KOTAKBANK.NS")
    stock1 = 'HDFCBANK.NS'
    stock2 = 'KOTAKBANK.NS'

X = train_data[stock1]
y = train_data[stock2]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
beta = model.params[stock1]

# 2. Construct the Spread
spread = train_data[stock2] - (beta * train_data[stock1])

# 3. Calculate Z-Score
mean_spread = spread.mean()
std_spread = spread.std()
z_score = (spread - mean_spread) / std_spread

# Display the first few signals
print(f"Hedge Ratio (Beta): {beta:.4f}")
print(z_score.tail())

def generate_signals(z_series, entry_z, exit_z, stop_z):
    pos = pd.Series(0, index=z_series.index)
    current_pos = 0
    
    for i in range(len(z_series)):
        z = z_series.iloc[i]
        
        # Check Exit / Stop Loss first if in position
        if current_pos != 0:
            # Stop Loss
            if (current_pos == 1 and z < -stop_z) or (current_pos == -1 and z > stop_z):
                current_pos = 0 # Exit with loss
            # Profit Take / Mean Reversion
            elif (current_pos == 1 and z >= -exit_z) or (current_pos == -1 and z <= exit_z):
                current_pos = 0 # Exit
                
        # Check Entry if not in position
        if current_pos == 0:
            if z < -entry_z:
                current_pos = 1 # Long Spread
            elif z > entry_z:
                current_pos = -1 # Short Spread
                
        pos.iloc[i] = current_pos
        
    return pos


# --- STEP 4: BACKTEST AND EVALUATE PERFORMANCE ---
intercept = model.params['const']  # This defines 'intercept' for your spread calculation
beta = model.params[stock1]

# 1. Strategy Parameters from Step 3
ENTRY_Z = 2.0  # Entry threshold
EXIT_Z = 0.0   # Mean reversion exit
STOP_Z = 4.0   # Stop loss threshold
TC_BPS = 5     # Transaction cost assumption (5 basis points)
tc = TC_BPS / 10000

# 2. Construct Out-of-Sample Signal (using Rolling Window for adaptivity)
test_spread = test_data[stock2] - (beta * test_data[stock1] + intercept)
# Using Rolling Statistics for Z-Score
window = 30
rolling_mean = test_spread.rolling(window=window).mean()
rolling_std = test_spread.rolling(window=window).std()
test_z = (test_spread - rolling_mean) / rolling_std
test_z = test_z.fillna(0)

# 3. Simulate Trading Logic
pos = generate_signals(test_z, ENTRY_Z, EXIT_Z, STOP_Z)

# 4. Calculate Net Returns (including Transaction Costs)
stock2_ret = test_data[stock2].pct_change()
stock1_ret = test_data[stock1].pct_change()

# Calculate trades to apply transaction costs
trades = pos.diff().abs().fillna(0)
# Net Daily Return = (Position * Spread Return) - Transaction Cost
daily_ret = (pos.shift(1) * (stock2_ret - beta * stock1_ret)) - (trades * tc)
cum_ret = (1 + daily_ret).cumprod()

# 5. Mandatory Metrics Calculation (as per image_8795bc.png)
def get_performance_metrics(returns, cum_returns):
    total_ret = cum_returns.iloc[-1] - 1
    ann_ret = (1 + total_ret)**(252/len(returns)) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / vol
    
    # Max Drawdown
    roll_max = cum_returns.cummax()
    drawdown = (cum_returns - roll_max) / roll_max
    mdd = drawdown.min()
    
    # Other PDF metrics
    num_trades = int(trades.sum())
    win_rate = len(returns[returns > 0]) / len(returns[returns != 0])
    calmar = ann_ret / abs(mdd)
    # Simple Sortino (downside vol)
    downside_ret = returns[returns < 0]
    sortino = ann_ret / (downside_ret.std() * np.sqrt(252))
    
    return {
        "CAGR": ann_ret, "Volatility": vol, "Sharpe": sharpe,
        "Max Drawdown": mdd, "Number of Trades": num_trades,
        "Win Rate": win_rate, "Calmar Ratio": calmar, "Sortino Ratio": sortino
    }

metrics = get_performance_metrics(daily_ret, cum_ret)

# --- DISPLAY RESULTS ---
print("\n--- MANDATORY PERFORMANCE METRICS ---")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")

# Draw Drawdown-time and Portfolio-value curves (Requirement from PDF)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Portfolio Value Curve
ax1.plot(cum_ret, color='blue', label='Strategy Equity Curve')
ax1.set_title('Portfolio Value with Time')
ax1.set_ylabel('Cumulative Wealth')
ax1.grid(True)

# Drawdown Curve
roll_max = cum_ret.cummax()
drawdown = (cum_ret - roll_max) / roll_max
ax2.fill_between(drawdown.index, drawdown, color='red', alpha=0.3, label='Drawdown')
ax2.set_title('Drawdown-Time Curve')
ax2.set_ylabel('Drawdown %')
ax2.grid(True)

plt.tight_layout()
plt.show()





# --- STEP 5: ROBUSTNESS CHECKS ---

def run_robustness_backtest(z_series, entry_threshold, tc_bps):
    """Simplified backtest function for rapid sensitivity testing."""
    tc = tc_bps / 10000
    pos = generate_signals(z_series, entry_threshold, exit_z=0.0, stop_z=4.0)
    
    # Calculate returns (using previously defined hdfc_ret and kotak_ret)
    # Calculate returns (using global stock returns)
    trades = pos.diff().abs().fillna(0)
    net_ret = (pos.shift(1) * (stock2_ret - beta * stock1_ret)) - (trades * tc)
    cum_ret = (1 + net_ret).cumprod()
    
    # Metrics
    ann_ret = (1 + (cum_ret.iloc[-1] - 1))**(252/len(net_ret)) - 1
    sharpe = (net_ret.mean() / net_ret.std()) * np.sqrt(252) if net_ret.std() != 0 else 0
    mdd = (cum_ret / cum_ret.cummax() - 1).min()
    
    return {'Entry_Z': entry_threshold, 'TC_bps': tc_bps, 
            'CAGR': ann_ret, 'Sharpe': sharpe, 'MaxDD': mdd}

# 1. Test Sensitivity to Z-Score Threshold (Stability check)
z_tests = [run_robustness_backtest(test_z, t, 5) for t in [1.5, 2.0, 2.5]]
z_robustness_df = pd.DataFrame(z_tests)

# 2. Test Sensitivity to Transaction Costs (Cost-fragility check)
tc_tests = [run_robustness_backtest(test_z, 2.0, c) for c in [5, 15, 30]]
tc_robustness_df = pd.DataFrame(tc_tests)

print("\n--- ROBUSTNESS CHECK 1: Z-SCORE THRESHOLD ---")
print(z_robustness_df)

print("\n--- ROBUSTNESS CHECK 2: TRANSACTION COSTS ---")
print(tc_robustness_df)

# Plotting Robustness for Report
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(z_robustness_df['Entry_Z'].astype(str), z_robustness_df['Sharpe'], color='skyblue')
plt.title('Sharpe vs Entry Z-Score')
plt.ylabel('Sharpe Ratio')

plt.subplot(1, 2, 2)
plt.plot(tc_robustness_df['TC_bps'], tc_robustness_df['CAGR'], marker='o', color='orange')
plt.title('CAGR vs Transaction Costs')
plt.xlabel('TC (bps)')
plt.ylabel('CAGR')
plt.tight_layout()
plt.show()


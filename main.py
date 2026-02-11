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
X = train_data['HDFCBANK.NS']
y = train_data['KOTAKBANK.NS']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
beta = model.params['HDFCBANK.NS']

# 2. Construct the Spread
spread = train_data['KOTAKBANK.NS'] - (beta * train_data['HDFCBANK.NS'])

# 3. Calculate Z-Score
mean_spread = spread.mean()
std_spread = spread.std()
z_score = (spread - mean_spread) / std_spread

# Display the first few signals
print(f"Hedge Ratio (Beta): {beta:.4f}")
print(z_score.tail())

# --- STEP 4: BACKTEST AND EVALUATE PERFORMANCE ---
intercept = model.params['const']  # This defines 'intercept' for your spread calculation
beta = model.params['HDFCBANK.NS']

# 1. Strategy Parameters from Step 3
ENTRY_Z = 2.0  # Entry threshold
EXIT_Z = 0.0   # Mean reversion exit
TC_BPS = 5     # Transaction cost assumption (5 basis points)
tc = TC_BPS / 10000

# 2. Construct Out-of-Sample Signal (using training mean/std)
test_spread = test_data['KOTAKBANK.NS'] - (beta * test_data['HDFCBANK.NS'] + intercept)
test_z = (test_spread - mean_spread) / std_spread

# 3. Simulate Trading Logic
pos = pd.Series(0, index=test_z.index)
pos[test_z < -ENTRY_Z] = 1   # Long Spread
pos[test_z > ENTRY_Z] = -1   # Short Spread

# Carry positions forward until exit threshold is crossed
pos = pos.replace(0, np.nan).ffill().fillna(0)
for i in range(1, len(test_z)):
    if pos.iloc[i-1] != 0 and abs(test_z.iloc[i]) <= EXIT_Z:
        pos.iloc[i:] = 0
        break

# 4. Calculate Net Returns (including Transaction Costs)
kotak_ret = test_data['KOTAKBANK.NS'].pct_change()
hdfc_ret = test_data['HDFCBANK.NS'].pct_change()

# Calculate trades to apply transaction costs
trades = pos.diff().abs().fillna(0)
# Net Daily Return = (Position * Spread Return) - Transaction Cost
daily_ret = (pos.shift(1) * (kotak_ret - beta * hdfc_ret)) - (trades * tc)
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


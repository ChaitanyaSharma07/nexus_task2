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
    
 
    adj_close_data = data['Adj Close']
    
    adj_close_data = adj_close_data.dropna(axis=1, how='all')
    
    return adj_close_data

all_data = load_data(nftybtickers, start, end)  

train_data = all_data.loc["2013-01-01":"2020-12-31"]

test_data = all_data.loc["2021-01-01":"2025-12-31"]

#generating combinations
pairs = list(combinations(nftybtickers, 2))
pairs_df = pd.DataFrame(pairs, columns=['Stock1', 'Stock2'])

#using correlation filter
def get_corr(row):
    return train_data[row['Stock1']].corr(train_data[row['Stock2']])

pairs_df['Correlation'] = pairs_df.apply(get_corr, axis=1)
filtered_pairs = pairs_df[pairs_df['Correlation'] > 0.90].copy()


def check_cointegration(row):
    S1 = train_data[row['Stock1']]
    S2 = train_data[row['Stock2']]
    
    #ols regression
    X = sm.add_constant(S1)
    model = sm.OLS(S2, X).fit()
    
    #engel granger test
    result = adfuller(model.resid)
    p_value = result[1]
    t_stat = result[0]
    
    return pd.Series([t_stat, p_value], index=['T-Stat', 'P-Value'])

results = filtered_pairs.apply(check_cointegration, axis=1)
final_report = pd.concat([filtered_pairs, results], axis=1)

#finding pairs based on p value
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

spread = train_data[stock2] - (beta * train_data[stock1])

mean_spread = spread.mean()
std_spread = spread.std()
z_score = (spread - mean_spread) / std_spread

print(f"Hedge Ratio (Beta): {beta:.4f}")
print(z_score.tail())

def generate_signals(z_series, entry_z, exit_z, stop_z):
    pos = pd.Series(0, index=z_series.index)
    current_pos = 0
    
    for i in range(len(z_series)):
        z = z_series.iloc[i]
        
        #
        if current_pos != 0:
            #stop loss~
            if (current_pos == 1 and z < -stop_z) or (current_pos == -1 and z > stop_z):
                current_pos = 0
            elif (current_pos == 1 and z >= -exit_z) or (current_pos == -1 and z <= exit_z):
                current_pos = 0 
    
        if current_pos == 0:
            if z < -entry_z:
                current_pos = 1 #long Spread
            elif z > entry_z:
                current_pos = -1 #short spread
                
        pos.iloc[i] = current_pos
        
    return pos


# backtesting
intercept = model.params['const']
beta = model.params[stock1]


ENTRY_Z = 2.0 
EXIT_Z = 0.0   
STOP_Z = 4.0   
TC_BPS = 5    
tc = TC_BPS / 10000

test_spread = test_data[stock2] - (beta * test_data[stock1] + intercept)
#finding z score with rolling window
window = 30
rolling_mean = test_spread.rolling(window=window).mean()
rolling_std = test_spread.rolling(window=window).std()
test_z = (test_spread - rolling_mean) / rolling_std
test_z = test_z.fillna(0)

pos = generate_signals(test_z, ENTRY_Z, EXIT_Z, STOP_Z)

#finding returns
stock2_ret = test_data[stock2].pct_change()
stock1_ret = test_data[stock1].pct_change()

trades = pos.diff().abs().fillna(0)
daily_ret = (pos.shift(1) * (stock2_ret - beta * stock1_ret)) - (trades * tc)
cum_ret = (1 + daily_ret).cumprod()

#finding metrics told
def get_performance_metrics(returns, cum_returns, manual_trade_count=None):
    total_ret = cum_returns.iloc[-1] - 1
    ann_ret = (1 + total_ret)**(252/len(returns)) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / vol
    
    #max drawdown
    roll_max = cum_returns.cummax()
    drawdown = (cum_returns - roll_max) / roll_max
    mdd = drawdown.min()
    
  
    if manual_trade_count is not None:
        num_trades = manual_trade_count
    else:
        num_trades = int(trades.sum())
        
    win_rate = len(returns[returns > 0]) / len(returns[returns != 0])
    calmar = ann_ret / abs(mdd)
    #sortino ratio
    downside_ret = returns[returns < 0]
    sortino = ann_ret / (downside_ret.std() * np.sqrt(252))
    
    return {
        "CAGR": ann_ret, "Volatility": vol, "Sharpe": sharpe,
        "Max Drawdown": mdd, "Number of Trades": num_trades,
        "Win Rate": win_rate, "Calmar Ratio": calmar, "Sortino Ratio": sortino
    }

metrics = get_performance_metrics(daily_ret, cum_ret)

#showing metrics
print("\n--- MANDATORY PERFORMANCE METRICS ---")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")

#drawing graphs
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
plt.tight_layout()
plt.show()

#comparing with benchmark

#nifty bank
print("\ndownloading nse bank data")
benchmark_df = yf.download("^NSEBANK", start=start, end=end, auto_adjust=False)

# Extract Adj Close properly handling potential MultiIndex or DataFrame output
if isinstance(benchmark_df.columns, pd.MultiIndex):
    benchmark_data = benchmark_df.xs('Adj Close', axis=1, level=0)
    if isinstance(benchmark_data, pd.DataFrame):
         benchmark_data = benchmark_data.iloc[:, 0]
elif 'Adj Close' in benchmark_df.columns:
    benchmark_data = benchmark_df['Adj Close']
else:
    benchmark_data = benchmark_df.iloc[:, 0] # Fallback

if isinstance(benchmark_data, pd.DataFrame):
    benchmark_data = benchmark_data.squeeze()

benchmark_test = benchmark_data.loc["2021-01-01":"2025-12-31"]

bench_ret = benchmark_test.pct_change().dropna()
bench_cum_ret = (1 + bench_ret).cumprod()

bench_metrics = get_performance_metrics(bench_ret, bench_cum_ret, manual_trade_count=1)
print("\n BENCHMARK PERFORMANCE (NIFTY BANK INDEX)")
for key, value in bench_metrics.items():
    print(f"{key}: {value:.4f}")

print(f"\n--- BUY & HOLD PERFORMANCE: {stock1} ---")
s1_ret = test_data[stock1].pct_change().dropna()
s1_cum_ret = (1 + s1_ret).cumprod()
s1_metrics = get_performance_metrics(s1_ret, s1_cum_ret, manual_trade_count=1)
for key, value in s1_metrics.items():
    print(f"{key}: {value:.4f}")

print(f"\n--- BUY & HOLD PERFORMANCE: {stock2} ---")
s2_ret = test_data[stock2].pct_change().dropna()
s2_cum_ret = (1 + s2_ret).cumprod()
s2_metrics = get_performance_metrics(s2_ret, s2_cum_ret, manual_trade_count=1)
for key, value in s2_metrics.items():
    print(f"{key}: {value:.4f}")


print(f"\nBUY & HOLD PERFORMANCE: EQUAL-WEIGHT PAIR ({stock1} + {stock2}) ---")
pair_ret = (s1_ret * 0.5) + (s2_ret * 0.5)
pair_cum_ret = (1 + pair_ret).cumprod()
pair_metrics = get_performance_metrics(pair_ret, pair_cum_ret, manual_trade_count=1)
for key, value in pair_metrics.items():
    print(f"{key}: {value:.4f}")

# 6. Plot Strategy vs Benchmarks (Comparison)
plt.figure(figsize=(12, 6))
plt.plot(cum_ret, label='Strategy Equity Curve', color='blue', linewidth=2)
plt.plot(bench_cum_ret, label='NIFTY Bank Index', color='gray', linestyle='--')
plt.plot(pair_cum_ret, label=f'Equal-Weight Buy&Hold ({stock1}+{stock2})', color='purple', linestyle='-.', alpha=0.8)

plt.title('Performance Comparison: Strategy vs Buy-and-Hold Benchmarks')
plt.ylabel('Cumulative Wealth')
plt.xlabel('Date')
plt.legend()
plt.grid(True)
plt.show()

#robustness checking

def run_robustness_backtest(z_series, entry_threshold, tc_bps):
    tc = tc_bps / 10000
    pos = generate_signals(z_series, entry_threshold, exit_z=0.0, stop_z=4.0)
    
    trades = pos.diff().abs().fillna(0)
    net_ret = (pos.shift(1) * (stock2_ret - beta * stock1_ret)) - (trades * tc)
    net_ret = net_ret.fillna(0)
    cum_ret = (1 + net_ret).cumprod()
    
    #metrics
    ann_ret = (1 + (cum_ret.iloc[-1] - 1))**(252/len(net_ret)) - 1
    sharpe = (net_ret.mean() / net_ret.std()) * np.sqrt(252) if net_ret.std() != 0 else 0
    mdd = (cum_ret / cum_ret.cummax() - 1).min()
    
    return {'Entry_Z': entry_threshold, 'TC_bps': tc_bps, 
            'CAGR': ann_ret, 'Sharpe': sharpe, 'MaxDD': mdd}

z_tests = [run_robustness_backtest(test_z, t, 5) for t in [1.5, 2.0, 2.5]]
z_robustness_df = pd.DataFrame(z_tests)

tc_tests = [run_robustness_backtest(test_z, 2.0, c) for c in [5, 15, 30]]
tc_robustness_df = pd.DataFrame(tc_tests)



mid_point = len(test_z) // 2
sub_period_1_z = test_z.iloc[:mid_point]
sub_period_2_z = test_z.iloc[mid_point:]

sub_1_res = run_robustness_backtest(sub_period_1_z, 2.0, 5)
sub_1_res['Period'] = 'First Half (2021-2023)'
sub_2_res = run_robustness_backtest(sub_period_2_z, 2.0, 5)
sub_2_res['Period'] = 'Second Half (2023-2025)'

sub_period_df = pd.DataFrame([sub_1_res, sub_2_res])

print("\nROBUSTNESS CHECK 1: Z-SCORE THRESHOLD")
print(z_robustness_df[['Entry_Z', 'CAGR', 'Sharpe', 'MaxDD']])

print("\nROBUSTNESS CHECK 2: TRANSACTION COSTS")
print(tc_robustness_df[['TC_bps', 'CAGR', 'Sharpe', 'MaxDD']])

print("\nROBUSTNESS CHECK 3: SUB-PERIOD ANALYSIS")
print(sub_period_df[['Period', 'CAGR', 'Sharpe', 'MaxDD']])

# Plotting Robustness for Report
plt.figure(figsize=(15, 5))

# Plot 1: Z-Score Sensitivity
plt.subplot(1, 3, 1)
plt.bar(z_robustness_df['Entry_Z'].astype(str), z_robustness_df['Sharpe'], color='skyblue')
plt.title('Sharpe vs Entry Z-Score')
plt.ylabel('Sharpe Ratio')
plt.xlabel('Z-Score Threshold')

# Plot 2: Transaction Cost Sensitivity
plt.subplot(1, 3, 2)
plt.plot(tc_robustness_df['TC_bps'], tc_robustness_df['CAGR'], marker='o', color='orange')
plt.title('CAGR vs Transaction Costs')
plt.xlabel('TC (bps)')
plt.ylabel('CAGR')

# Plot 3: Sub-Period Analysis
plt.subplot(1, 3, 3)
plt.bar(sub_period_df['Period'], sub_period_df['Sharpe'], color=['lightgreen', 'salmon'])
plt.title('Sharpe Ratio by Sub-Period')
plt.ylabel('Sharpe Ratio')
plt.xticks(rotation=15)

plt.tight_layout()
plt.show()


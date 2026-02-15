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
intercept = model.params['const']

# 1. Calculate the Spread using the training-derived Beta and Intercept
# Formula: S2 - (Beta * S1 + Intercept)
spread = test_data[stock2] - (beta * test_data[stock1] + intercept)

# 2. Create the Plot
plt.figure(figsize=(14, 6))
plt.plot(spread, color='teal', linewidth=1.5, label=f'Spread ({stock2} - {beta:.2f} * {stock1})')

# 3. Add the Mean Line (the equilibrium)
plt.axhline(spread.mean(), color='black', linestyle='--', linewidth=1, label='Mean (Equilibrium)')

# 4. Add Standard Deviation Bands (Visualizing the "Stretch")
plt.axhline(spread.mean() + spread.std(), color='red', linestyle=':', alpha=0.5, label='1 Std Dev')
plt.axhline(spread.mean() - spread.std(), color='red', linestyle=':', alpha=0.5)

# 5. Formatting for the Research Report
plt.title(f'The Engle-Granger Spread: {stock1} vs {stock2}', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Spread Value', fontsize=12)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

plt.tight_layout()

normalized_stock1 = (test_data[stock1] / test_data[stock1].iloc[0]) * 100
normalized_stock2 = (test_data[stock2] / test_data[stock2].iloc[0]) * 100

# 2. Plotting
plt.figure(figsize=(14, 7))
plt.plot(normalized_stock1, label=f'{stock1} (Normalized)', color='blue', alpha=0.8)
plt.plot(normalized_stock2, label=f'{stock2} (Normalized)', color='orange', alpha=0.8)

# 3. Formatting for the Research Report
plt.title(f'Price Correlation: {stock1} vs {stock2} (Normalized to 100)', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Normalized Price Index', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
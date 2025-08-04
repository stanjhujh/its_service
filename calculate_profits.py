import numpy as np
from datetime import datetime

# Calculate Base Profit from already descaled data
def calculate_profits(df, MIN_PRICE_RES, S_PER_POINT, MAX_DRAWDOWN):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Calculating Base Profit from descaled data...")
    print(df.shape)
    # Since data is already descaled, we can use it directly
    data = df.iloc[:, 1:].values  # Exclude Date
    
    long_profits = np.zeros(len(df))
    short_profits = np.zeros(len(df))
    max_long_profits = np.zeros(len(df))
    max_short_profits = np.zeros(len(df))
    df['TradingDay'] = df['Date'].dt.date
    valid_indices = []
    
    for i in range(len(df) - 6):
        if not df['InWindow'].iloc[i]:
            continue
        current_day = df['TradingDay'].iloc[i]
        if i + 6 < len(df) and df['TradingDay'].iloc[i + 1:i + 7].eq(current_day).all():
            valid_indices.append(i)
            open_t1 = data[i + 1, df.columns.get_loc('Open') - 1]
            close_t6 = data[i + 6, df.columns.get_loc('Close') - 1]
            lows_t1_t6 = data[i + 1:i + 7, df.columns.get_loc('Low') - 1]
            highs_t1_t6 = data[i + 1:i + 7, df.columns.get_loc('High') - 1]
            
            # Base Long Profit
            long_points = (close_t6 - open_t1) / MIN_PRICE_RES
            long_profit = long_points * (S_PER_POINT * MIN_PRICE_RES)
            long_drawdown = ((open_t1 - min(lows_t1_t6)) / MIN_PRICE_RES) * (S_PER_POINT * MIN_PRICE_RES)
            long_profits[i] = -MAX_DRAWDOWN if long_drawdown >= MAX_DRAWDOWN else long_profit
            
            # Base Short Profit
            short_points = (open_t1 - close_t6) / MIN_PRICE_RES
            short_profit = short_points * (S_PER_POINT * MIN_PRICE_RES)
            short_drawdown = ((max(highs_t1_t6) - open_t1) / MIN_PRICE_RES) * (S_PER_POINT * MIN_PRICE_RES)
            short_profits[i] = -MAX_DRAWDOWN if short_drawdown >= MAX_DRAWDOWN else short_profit
            
            # Maximum Long Profit
            if long_drawdown >= MAX_DRAWDOWN:
                drawdown_bar = np.argmin(lows_t1_t6) + 1
                max_long_profit = ((max(highs_t1_t6[:drawdown_bar]) - open_t1) / MIN_PRICE_RES) * (S_PER_POINT * MIN_PRICE_RES)
                max_long_profits[i] = max_long_profit
            else:
                max_long_profit = ((max(highs_t1_t6) - open_t1) / MIN_PRICE_RES) * (S_PER_POINT * MIN_PRICE_RES)
                max_long_profits[i] = max_long_profit
            
            # Maximum Short Profit
            if short_drawdown >= MAX_DRAWDOWN:
                drawdown_bar = np.argmax(highs_t1_t6) + 1
                max_short_profit = ((open_t1 - min(lows_t1_t6[:drawdown_bar])) / MIN_PRICE_RES) * (S_PER_POINT * MIN_PRICE_RES)
                max_short_profits[i] = max_short_profit
            else:
                max_short_profit = ((open_t1 - min(lows_t1_t6)) / MIN_PRICE_RES) * (S_PER_POINT * MIN_PRICE_RES)
                max_short_profits[i] = max_short_profit
#         print("i: ",i)
#         print("valid_indices: ",valid_indices[-1])
    
    valid_indices = np.array(valid_indices)
#     print()
    print(f"Valid Profit Indices: {len(valid_indices)} bars with non-zero profits\n")
    
    return long_profits, short_profits, valid_indices
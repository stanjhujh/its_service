import numpy as np
from datetime import datetime

# Calculate Base Profit from already descaled data
def calculate_profits(df, MIN_PRICE_RES, S_PER_POINT, TARGET_PROFIT, STOP_LOSS):
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
            
            # Dynamic exit logic - check each bar for stop loss/target profit
            long_exit_bar = 6  # Default to hard exit at 6th bar
            short_exit_bar = 6  # Default to hard exit at 6th bar
            
            # Check for early exits due to stop loss or target profit
            for j in range(1, 7):
                if i + j >= len(df):
                    break
                    
                close_tj = data[i + j, df.columns.get_loc('Close') - 1]
                low_tj = data[i + j, df.columns.get_loc('Low') - 1]
                high_tj = data[i + j, df.columns.get_loc('High') - 1]
                
                # Check long position drawdown (stop loss)
                long_drawdown = ((open_t1 - low_tj) / MIN_PRICE_RES) * (S_PER_POINT * MIN_PRICE_RES)
                if long_drawdown >= STOP_LOSS and long_exit_bar == 6:
                    long_exit_bar = j
                
                # Check long position profit (target profit)
                long_profit_at_close = ((close_tj - open_t1) / MIN_PRICE_RES) * (S_PER_POINT * MIN_PRICE_RES)
                if long_profit_at_close >= TARGET_PROFIT and long_exit_bar == 6:
                    long_exit_bar = j
                
                # Check short position drawdown (stop loss)
                short_drawdown = ((high_tj - open_t1) / MIN_PRICE_RES) * (S_PER_POINT * MIN_PRICE_RES)
                if short_drawdown >= STOP_LOSS and short_exit_bar == 6:
                    short_exit_bar = j
                
                # Check short position profit (target profit)
                short_profit_at_close = ((open_t1 - close_tj) / MIN_PRICE_RES) * (S_PER_POINT * MIN_PRICE_RES)
                if short_profit_at_close >= TARGET_PROFIT and short_exit_bar == 6:
                    short_exit_bar = j
            
            # Calculate final profits based on exit bars
            close_long_exit = data[i + long_exit_bar, df.columns.get_loc('Close') - 1]
            close_short_exit = data[i + short_exit_bar, df.columns.get_loc('Close') - 1]
            
            # Long profit calculation
            long_points = (close_long_exit - open_t1) / MIN_PRICE_RES
            long_profit = long_points * (S_PER_POINT * MIN_PRICE_RES)
            lows_to_exit = data[i + 1:i + long_exit_bar + 1, df.columns.get_loc('Low') - 1]
            long_drawdown = ((open_t1 - min(lows_to_exit)) / MIN_PRICE_RES) * (S_PER_POINT * MIN_PRICE_RES)
            long_profits[i] = -STOP_LOSS if long_drawdown >= STOP_LOSS else min(long_profit, TARGET_PROFIT)
            
            # Short profit calculation
            short_points = (open_t1 - close_short_exit) / MIN_PRICE_RES
            short_profit = short_points * (S_PER_POINT * MIN_PRICE_RES)
            highs_to_exit = data[i + 1:i + short_exit_bar + 1, df.columns.get_loc('High') - 1]
            short_drawdown = ((max(highs_to_exit) - open_t1) / MIN_PRICE_RES) * (S_PER_POINT * MIN_PRICE_RES)
            short_profits[i] = -STOP_LOSS if short_drawdown >= STOP_LOSS else min(short_profit, TARGET_PROFIT)
            
            # Maximum profits (unchanged logic)
            lows_t1_t6 = data[i + 1:i + 7, df.columns.get_loc('Low') - 1]
            highs_t1_t6 = data[i + 1:i + 7, df.columns.get_loc('High') - 1]
            
            # Maximum Long Profit
            if long_drawdown >= STOP_LOSS:
                drawdown_bar = np.argmin(lows_to_exit) + 1
                max_long_profit = ((max(highs_t1_t6[:drawdown_bar]) - open_t1) / MIN_PRICE_RES) * (S_PER_POINT * MIN_PRICE_RES)
                max_long_profits[i] = max_long_profit
            else:
                max_long_profit = ((max(highs_t1_t6) - open_t1) / MIN_PRICE_RES) * (S_PER_POINT * MIN_PRICE_RES)
                max_long_profits[i] = max_long_profit
            
            # Maximum Short Profit
            if short_drawdown >= STOP_LOSS:
                drawdown_bar = np.argmax(highs_to_exit) + 1
                max_short_profit = ((open_t1 - min(lows_t1_t6[:drawdown_bar])) / MIN_PRICE_RES) * (S_PER_POINT * MIN_PRICE_RES)
                max_short_profits[i] = max_short_profit
            else:
                max_short_profit = ((open_t1 - min(lows_t1_t6)) / MIN_PRICE_RES) * (S_PER_POINT * MIN_PRICE_RES)
                max_short_profits[i] = max_short_profit
    
    valid_indices = np.array(valid_indices)
    print(f"Valid Profit Indices: {len(valid_indices)} bars with non-zero profits\n")
    
    return long_profits, short_profits, valid_indices
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from calculate_profits import calculate_profits

def create_test_data():
    """Create sample data to test dynamic exits"""
    # Create sample OHLC data
    dates = []
    opens = []
    highs = []
    lows = []
    closes = []
    in_window = []
    
    # Create 10 bars of test data
    base_price = 6021.25
    base_time = datetime(2025, 7, 20, 9, 12, 40)
    
    for i in range(10):
        date = base_time + timedelta(seconds=i*5)
        dates.append(date)
        
        # Create different scenarios based on user's example
        if i == 0:  # Signal bar (09:12:40)
            opens.append(base_price)
            highs.append(base_price + 0.5)
            lows.append(base_price - 0.25)
            closes.append(base_price + 0.25)
            in_window.append(True)
        elif i == 1:  # Bar 1 - Entry bar (09:12:45) - Entry at 6021.25
            opens.append(base_price)  # Entry price
            highs.append(base_price + 0.75)
            lows.append(base_price - 0.1)
            closes.append(base_price + 0.5)
            in_window.append(False)
        elif i == 2:  # Bar 2 (09:12:50)
            opens.append(base_price + 0.5)
            highs.append(base_price + 1.5)
            lows.append(base_price + 0.4)
            closes.append(base_price + 1.0)
            in_window.append(False)
        elif i == 3:  # Bar 3 (09:12:55) - TP hit scenario
            opens.append(base_price + 1.0)
            highs.append(base_price + 2.5)
            lows.append(base_price + 0.9)
            closes.append(base_price + 2.0)  # Close at 6023.25 (TP hit: $100 profit)
            in_window.append(False)
        elif i == 4:  # Bar 4 (09:13:00)
            opens.append(base_price + 2.0)
            highs.append(base_price + 2.1)
            lows.append(base_price + 1.9)
            closes.append(base_price + 2.05)
            in_window.append(False)
        elif i == 5:  # Bar 5 (09:13:05)
            opens.append(base_price + 2.05)
            highs.append(base_price + 2.2)
            lows.append(base_price + 2.0)
            closes.append(base_price + 2.1)
            in_window.append(False)
        elif i == 6:  # Bar 6 (09:13:10) - Hard exit
            opens.append(base_price + 2.1)
            highs.append(base_price + 2.3)
            lows.append(base_price + 2.0)
            closes.append(base_price + 2.2)  # Close at 6023.45
            in_window.append(False)
        else:  # Remaining bars
            opens.append(base_price + 2.2)
            highs.append(base_price + 2.4)
            lows.append(base_price + 2.1)
            closes.append(base_price + 2.3)
            in_window.append(False)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'InWindow': in_window
    })
    
    return df

def test_dynamic_exits():
    """Test the dynamic exit logic"""
    print("Testing Dynamic Exit Logic...")
    print("=" * 50)
    
    # Create test data
    df = create_test_data()
    print("Test Data:")
    print(df[['Date', 'Open', 'High', 'Low', 'Close', 'InWindow']].to_string())
    print()
    
    # Test parameters
    MIN_PRICE_RES = 0.25
    S_PER_POINT = 50.0
    TARGET_PROFIT = 100.0  # $100 target
    STOP_LOSS = 150.0      # $150 stop loss
    
    print(f"Parameters:")
    print(f"  Target Profit: ${TARGET_PROFIT}")
    print(f"  Stop Loss: ${STOP_LOSS}")
    print(f"  Price per point: ${S_PER_POINT}")
    print(f"  Min price resolution: {MIN_PRICE_RES}")
    print()
    
    # Debug: Check the exact values being used
    entry_price = df.iloc[1]['Open']  # Bar 1 Open
    bar3_close = df.iloc[3]['Close']  # Bar 3 Close
    print(f"Debug Values:")
    print(f"  Entry price (Bar 1 Open): {entry_price}")
    print(f"  Bar 3 Close: {bar3_close}")
    print(f"  Price difference: {bar3_close - entry_price}")
    print(f"  Manual profit: {(bar3_close - entry_price) * S_PER_POINT}")
    print(f"  Should hit TP: {(bar3_close - entry_price) * S_PER_POINT >= TARGET_PROFIT}")
    print()
    
    # Calculate profits
    long_profits, short_profits, valid_indices = calculate_profits(
        df, MIN_PRICE_RES, S_PER_POINT, TARGET_PROFIT, STOP_LOSS
    )
    
    # Analyze results
    print("Results:")
    print(f"  Valid indices: {valid_indices}")
    print(f"  Long profits: {long_profits[valid_indices]}")
    print(f"  Short profits: {short_profits[valid_indices]}")
    
    # Expected results for our test data:
    # Entry at Bar 1 Open: 6021.25
    # Bar 3 Close: 6023.25
    # Long profit = (6023.25 - 6021.25) * 50 = $100.00 (TP hit)
    
    print()
    print("Expected vs Actual:")
    print("  Expected Long Profit: $100.00 (TP hit at Bar 3)")
    print("  Expected Short Profit: Negative (no TP hit)")
    
    # Manual calculation verification
    entry_price = 6021.25
    exit_price = 6023.25
    manual_profit = (exit_price - entry_price) * 50
    print(f"  Manual calculation: ({exit_price} - {entry_price}) * 50 = ${manual_profit}")
    
    return df, long_profits, short_profits, valid_indices

if __name__ == "__main__":
    test_dynamic_exits() 
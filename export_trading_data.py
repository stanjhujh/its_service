import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json

def export_trading_data(df, feature_data_test, test_labels, test_valid_indices, test_df, 
                       xgb_model, rf_model, DIRECTORIES, SYMBOL, SUFFIX, VERSION):
    """
    Export trading data to CSV with the exact structure requested:
    1. Raw OHLC data: Datetime, Open, High, Low, Close (for all signal rows including NO_TRADE)
    2. Trade signals based on model predictions: Trade Signal, EntryDateTime, EntryPrice, ExitDateTime, ExitPrice, BaseProfit
    """
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Exporting trading data to CSV...")
    
    # Create output directory if it doesn't exist
    output_dir = Path('./output')
    output_dir.mkdir(exist_ok=True)
    
    # Load asset parameters for proper profit calculation
    with open('user_parameters.json', 'r') as file:
        data = json.load(file)
    asset_data = data['assets'][SYMBOL]
    S_PER_POINT = asset_data['price_per_point']
    
    # ===== EXPORT 1: RAW OHLC DATA (FOR ALL SIGNAL ROWS) =====
    print("Exporting raw OHLC data for all signal rows...")
    
    # Get model predictions first
    xgb_probs = xgb_model.predict_proba(feature_data_test)
    rf_probs = rf_model.predict_proba(feature_data_test)
    ensemble_probs = 0.7 * xgb_probs + 0.3 * rf_probs
    
    # Generate predictions
    predicted_actions = np.full(len(feature_data_test), 2)  # Default to No Trade
    
    for i in range(len(ensemble_probs)):
        long_prob, short_prob, no_trade_prob = ensemble_probs[i]
        
        if (long_prob > short_prob and long_prob > no_trade_prob):  
            predicted_actions[i] = 0  # Long
        elif (short_prob >= long_prob and short_prob > no_trade_prob):
            predicted_actions[i] = 1  # Short
    
    # Collect indices for all signal rows (including NO_TRADE)
    all_signal_indices = []
    signal_names = {0: 'LONG', 1: 'SHORT', 2: 'NO_TRADE'}
    
    for i in range(len(feature_data_test)):
        test_idx = test_valid_indices[i] if i < len(test_valid_indices) else 0
        
        # Skip if not enough future data
        if test_idx + 6 >= len(df):
            continue
            
        # Include all signals (LONG, SHORT, NO_TRADE)
        all_signal_indices.append(test_idx)
    
    # Extract OHLC data for all signal rows
    if all_signal_indices:
        ohlc_data = df.iloc[all_signal_indices][['Date', 'Open', 'High', 'Low', 'Close']].copy()
        ohlc_data.columns = ['Datetime', 'Open', 'High', 'Low', 'Close']
        
        # Save raw OHLC data
        ohlc_filename = f'./output/{SYMBOL}_raw_ohlc_data{SUFFIX}{VERSION}.csv'
        ohlc_data.to_csv(ohlc_filename, index=False)
        print(f"Raw OHLC data saved to: {ohlc_filename}")
        print(f"Records: {len(ohlc_data)} (all signal rows including NO_TRADE)")
    else:
        print("No signal rows found - no OHLC data to export")
        ohlc_filename = None
    
    # ===== EXPORT 2: TRADE SIGNALS BASED ON MODEL PREDICTIONS =====
    print("Exporting trade signals based on model predictions...")
    
    # Create trade signals DataFrame
    trade_signals = []
    
    # Process each test prediction
    for i in range(len(feature_data_test)):
        test_idx = test_valid_indices[i] if i < len(test_valid_indices) else 0
        
        # Skip if not enough future data
        if test_idx + 6 >= len(df):
            continue
            
        # Get predicted signal
        predicted_signal = signal_names.get(predicted_actions[i], 'NO_TRADE')
        
        # Entry occurs at the OPEN of the NEXT bar (not current bar)
        entry_datetime = df.iloc[test_idx + 1]['Date']  # Next bar
        entry_price = df.iloc[test_idx + 1]['Open']     # Open of next bar
        
        # Exit occurs 6 bars AFTER the entry bar
        exit_datetime = df.iloc[test_idx + 6]['Date']   # 5 bars after entry
        exit_price = df.iloc[test_idx + 6]['Close']     # Close of 5th bar
        
        # Calculate base profit with proper multiplier
        if predicted_signal == 'LONG':
            base_profit = (exit_price - entry_price) * S_PER_POINT
        elif predicted_signal == 'SHORT':
            base_profit = (entry_price - exit_price) * S_PER_POINT
        else:  # NO_TRADE
            base_profit = 0  # No profit for NO_TRADE
            
        trade_signals.append({
            'Trade Signal': predicted_signal,
            'EntryDateTime': entry_datetime,
            'EntryPrice': entry_price,
            'ExitDateTime': exit_datetime,
            'ExitPrice': exit_price,
            'BaseProfit': base_profit
        })
    
    # Create DataFrame and save
    if trade_signals:
        trade_df = pd.DataFrame(trade_signals)
        trade_filename = f'./output/{SYMBOL}_trade_signals{SUFFIX}{VERSION}.csv'
        trade_df.to_csv(trade_filename, index=False)
        print(f"Trade signals saved to: {trade_filename}")
        print(f"Trade records: {len(trade_df)}")
        
        # Print signal distribution
        signal_counts = trade_df['Trade Signal'].value_counts()
        print(f"Signal distribution: {dict(signal_counts)}")
    else:
        print("No trade signals generated")
        trade_filename = None
    
    print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Data export completed!")
    
    return {
        'ohlc_filename': ohlc_filename,
        'trade_filename': trade_filename,
        'total_ohlc_records': len(all_signal_indices) if all_signal_indices else 0,
        'total_trades': len(trade_signals) if trade_signals else 0
    }

if __name__ == "__main__":
    print("Trading data export module loaded successfully!") 
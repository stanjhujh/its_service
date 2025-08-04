import pandas as pd
from datetime import datetime

def split_data(df):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Splitting data by custom cutoff and trading day...")

    # Ensure TradingDay column exists
    df['TradingDay'] = df['Date'].dt.date

    # Define test set cutoff
    test_start_date = pd.Timestamp('2025-07-07 08:30:00')
    
    # Split into train+val and test
    train_val_df = df[df['Date'] < test_start_date]
    test_df = df[df['Date'] >= test_start_date]
    
    # Split train+val by trading day
    trading_days = sorted(train_val_df['TradingDay'].unique())
    n_days = len(trading_days)
    train_days = trading_days[:int(0.8 * n_days)]
    val_days = trading_days[int(0.8 * n_days):]

    
    train_df = train_val_df[train_val_df['TradingDay'].isin(train_days)]
    val_df = train_val_df[train_val_df['TradingDay'].isin(val_days)]
    
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Data split completed:")
    print(test_df.head())
    print(f"  → Train: {len(train_df)} rows")
    print(f"  → Validation: {len(val_df)} rows")
    print(f"  → Test: {len(test_df)} rows starting from {test_df['Date'].iloc[0] if not test_df.empty else 'N/A'}")
    
    return train_df, val_df, test_df

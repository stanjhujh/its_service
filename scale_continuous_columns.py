from datetime import datetime

# Scale continuous columns in the dataframe
def scale_continuous_columns(df, scaler):
    """Scale the continuous columns in the dataframe using the provided scaler"""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Scaling continuous columns...")
    
    continuous_cols = [
        'Open', 'High', 'Low', 'Close', 'RSI', 'MACD', 'HABodyRangeRatio', 'MyWAZLTTrend',
        'Open_lag1', 'High_lag1', 'Low_lag1', 'Close_lag1', 'RSI_lag1', 'MACD_lag1', 'HABodyRangeRatio_lag1', 'MyWAZLTTrend_lag1'
    ]
    # Only keep columns that exist in df
    continuous_cols = [col for col in continuous_cols if col in df.columns]
    
    if not continuous_cols or scaler is None:
        print("Warning: No continuous columns to scale or scaler is None\n")
        return df
    
    # Create a copy of the dataframe
    df_scaled = df.copy()

    print(df_scaled.head())
    
    # Scale the continuous columns
    df_scaled[continuous_cols] = scaler.transform(df[continuous_cols])
    
    print(f"Continuous columns scaled successfully: {', '.join(continuous_cols)}\n")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Continuous columns scaled successfully.")
    
    return df_scaled
import pandas as pd
from datetime import datetime

# Load and validate data
def load_data(DIRECTORIES):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Loading and validating data...")
    dtypes = {col: 'float64' for col in pd.read_csv(DIRECTORIES['PREPROCESSED_FILE'], nrows=0).columns[1:]}
    dtypes['Date'] = 'str'
    df = pd.read_csv(DIRECTORIES['PREPROCESSED_FILE'], dtype=dtypes)
    print(f"## Data Validation\nLoaded {len(df)} rows from {DIRECTORIES['PREPROCESSED_FILE']}\n")
    
    n_cols = len(df.columns)
    print(f"Total column count: {n_cols}\n")
    # print(f"Column names: {', '.join(df.columns)}\n")
    
    if n_cols != 333:
        error_msg = f"Expected 333 columns, found {n_cols}"
        print(error_msg)
        raise ValueError(error_msg)
    
    date_formats = ['%Y-%m-%d %H:%M:%S', '%m/%d/%Y %I:%M:%S %p', '%m/%d/%y %I:%M:%S %p']
    parsed_dates = None
    for fmt in date_formats:
        try:
            parsed_dates = pd.to_datetime(df['Date'], format=fmt, errors='coerce')
            if parsed_dates.notna().any():
                print(f"Date parsing succeeded with format: {fmt}\n")
                break
        except Exception:
            continue
    
    if parsed_dates is None or parsed_dates.isna().all():
        try:
            parsed_dates = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
            print("Date parsing succeeded with mixed format inference.\n")
        except Exception as e:
            error_msg = f"Failed to parse datetime: {str(e)}"
            print(error_msg)
            raise ValueError(error_msg)
    
    df['Date'] = parsed_dates
    invalid_dates = df['Date'].isna().sum()
    if invalid_dates > 0:
        print(f"Dropped {invalid_dates} rows with invalid dates.\n")
        df = df.dropna(subset=['Date'])
    
    print(f"Remaining rows after date validation: {len(df)}\n")
    
    seconds = df['Date'].dt.second
    invalid_precision = (seconds % 5 != 0).sum()
    if invalid_precision > 0:
        print(f"Found {invalid_precision} rows with non-5-second precision; dropping.\n")
        df = df[seconds % 5 == 0]
    
    print(f"Remaining rows after precision check: {len(df)}\n")
    
    df['Time'] = df['Date'].dt.time
    df['InWindow'] = df['Time'].apply(lambda x: pd.Timestamp('08:35:00').time() <= x <= pd.Timestamp('10:25:00').time())
    print(f"Bars in trading window (08:35–10:25): {df['InWindow'].sum()}\n")
    
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Data loaded and validated: {len(df)} rows.")
#     print(df)
    return df

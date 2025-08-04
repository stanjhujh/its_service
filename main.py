import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
from xgboost import XGBClassifier
from pathlib import Path
from datetime import datetime
import json

# Import custom functions
from validate_files import validate_files
from load_data import load_data
from calculate_profits import calculate_profits
from scale_continuous_columns import scale_continuous_columns
from split_data import split_data
from select_imp_features import select_important_features
from generate_action_labels import generate_action_labels
from evaluate_model import evaluate_model
from export_trading_data import export_trading_data



# User Provided Parameters - Futures Configuration
SYMBOL = "ES"
K = 0.2

# # Suffix for all output files (e.g., '_v3'). Change as needed.
SUFFIX = "_0825"
VERSION = "_v10"

with open('user_parameters.json', 'r') as file:
    data = json.load(file)

asset_data = data['assets'][SYMBOL]

# Load values into individual variables
S_PER_POINT = asset_data['price_per_point']                    
MIN_PRICE_RES = asset_data['minimum_resolution_points']       
PRICE_PER_TICK = asset_data['price_per_tick']                 
MAX_PENALTY = asset_data['maximum_drawdown']                   
MAX_DRAWDOWN = asset_data['maximum_drawdown']                  
POINTS_TO_MAX_DRAWDOWN = asset_data['points_to_maximum_drawdown']

print(asset_data)



DIRECTORIES = {
    'DATA_DIR': '../data',
    'OUTPUT_DIR': '../output',
    'CONFIG_DIR': '../config',
    'PREPROCESSED_FILE': 'data/ES_preprocessed_all_features_training_zeros_corrected.csv',
    'XGB_TRAINED_MODEL_FILE': f'./output/{SYMBOL}_XGB_model{SUFFIX}{VERSION}.pkl',
    'RF_TRAINED_MODEL_FILE': f'./output/{SYMBOL}_RF_model{SUFFIX}{VERSION}.pkl',
    'DOCUMENTATION_FILE': f'./output/{SYMBOL}_model_documentation{SUFFIX}{VERSION}.md',
    # 'SYMBOL_SCALER_FILE': 'data/scalers/ES_scaler.pkl'
}


with open('selected_features.json', 'r') as file:
    selected_features = json.load(file)

SELECTED_FEATURES = selected_features['selected_features']

print(SELECTED_FEATURES)


"""
Main Execution Block
Initiates the model training process by logging the start time. Calls `init_documentation` to create a  file with model metadata. 
Validates input files using `validate_files` to ensure data availability. 
Loads and preprocesses data with `load_data`, returning a cleaned DataFrame.
"""

print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Starting model training script execution...")
validate_files(DIRECTORIES)
df = load_data(DIRECTORIES)


"""
Profit Calculation from Descaled Data
Computes long and short profits over a 6-bar horizon using descaled price data. 
Calculates base profits and maximum profits, applying a $150 drawdown penalty if exceeded. 
Returns arrays of long profits, short profits, and valid indices.
"""

# Calculate profits from descaled data
long_profits, short_profits, valid_indices = calculate_profits(df, MIN_PRICE_RES, S_PER_POINT, MAX_DRAWDOWN)


"""
Scale Continuous Columns
Applies the provided `StandardScaler` to normalize continuous columns (e.g., Open, High, Low, Close, RSI) in the DataFrame. 
Creates a copy of the DataFrame to avoid modifying the original data. Logs the scaled columns for verification. 
Returns the scaled DataFrame for model training.
"""
scaler = joblib.load('./data/ES_scaler.pkl')

# Scale the continuous columns for training
df_scaled = scale_continuous_columns(df, scaler)


"""
Scaling Verification
Verifies that the scaling of continuous columns (e.g., Open, High, Low, Close, RSI) was successful. 
Logs the mean and standard deviation of these columns before and after scaling. 
Ensures that post-scaling, the mean is approximately 0 and the standard deviation is approximately 1. 
Writes results to the documentation file for transparency.
"""

# Verify scaling worked correctly
continuous_cols = ['Open', 'High', 'Low', 'Close', 'RSI', 'MACD', 'HABodyRangeRatio', 'MyWAZLTTrend']
continuous_cols = [col for col in continuous_cols if col in df.columns]

print("## Scaling Verification\n")
print("Before scaling (descaled data):\n")
for col in continuous_cols:
    col_mean = df[col].mean()
    col_std = df[col].std()
    print(f"{col}: mean={col_mean:.4f}, std={col_std:.4f}\n")

print("After scaling:\n")
for col in continuous_cols:
    col_mean = df_scaled[col].mean()
    col_std = df_scaled[col].std()
    print(f"{col}: mean={col_mean:.4f}, std={col_std:.4f}\n")

print("Note: After scaling, continuous columns should have mean≈0 and std≈1\n")


"""
Data Splitting
Splits the scaled DataFrame into training (70%), validation (20%), and test (10%) sets based on trading days. 
Ensures temporal consistency by assigning entire trading days to each set.
Returns the training, validation, and test DataFrames.
"""

train_df, val_df, test_df = split_data(df_scaled)


"""
Feature Selection and Data Preparation
Selects 76 predefined features from the scaled DataFrame and validates their presence, 
raising errors if any are missing or the count is incorrect. Extracts feature data for training, validation, and test sets, 
ensuring exactly 76 features are used. Filters indices to include only those with valid profit calculations. 
Logs feature details and data shapes for documentation.
"""

# TRAIN MODEL (WITH PRICE FEATURES)
print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Training model with price features...")

# Feature selection (76 features including price features)
important_features = select_important_features(df_scaled)

# Validate that we got exactly 76 features
if len(important_features) != 76:
    error_msg = f"ERROR: Expected 76 features, but got {len(important_features)}"
    print(error_msg)
    raise ValueError(error_msg)

# Get column indices for the selected features
feature_columns = [df_scaled.columns.get_loc(f) for f in important_features if f in df_scaled.columns]

# Validate that all features were found in the dataframe
if len(feature_columns) != len(important_features):
    missing_features = [f for f in important_features if f not in df_scaled.columns]
    error_msg = f"ERROR: Some features not found in dataframe: {missing_features}"
    print(error_msg)
    raise ValueError(error_msg)

print(f"Feature Validation: Successfully selected {len(important_features)} features\n")

# Get all features first, then select important ones
all_feature_data_train = train_df.iloc[:, 1:333].values
all_feature_data_val = val_df.iloc[:, 1:333].values
all_feature_data_test = test_df.iloc[:, 1:333].values

# Select only important features
feature_data_train = all_feature_data_train[:, [i-1 for i in feature_columns]]
feature_data_val = all_feature_data_val[:, [i-1 for i in feature_columns]]
feature_data_test = all_feature_data_test[:, [i-1 for i in feature_columns]]

# Validate feature data shapes
expected_features = 76
if feature_data_train.shape[1] != expected_features:
    error_msg = f"ERROR: Training data has {feature_data_train.shape[1]} features, expected {expected_features}"
    print(error_msg)
    raise ValueError(error_msg)

if feature_data_val.shape[1] != expected_features:
    error_msg = f"ERROR: Validation data has {feature_data_val.shape[1]} features, expected {expected_features}"
    print(error_msg)
    raise ValueError(error_msg)

if feature_data_test.shape[1] != expected_features:
    error_msg = f"ERROR: Test data has {feature_data_test.shape[1]} features, expected {expected_features}"
    print(error_msg)
    raise ValueError(error_msg)

print(f"Successfully using {feature_data_train.shape[1]} selected features (including price features)\n")
print(f"Data Shapes - Train: {feature_data_train.shape}, Val: {feature_data_val.shape}, Test: {feature_data_test.shape}\n")

train_indices = train_df.index.values
val_indices = val_df.index.values
test_indices = test_df.index.values #7260

# Filter indices to valid profits
train_valid_mask = np.isin(train_indices, valid_indices)
train_valid_indices = train_indices[train_valid_mask]
feature_data_train = feature_data_train[train_valid_mask]
train_df_filtered = train_df.iloc[train_valid_mask]

val_valid_mask = np.isin(val_indices, valid_indices)
val_valid_indices = val_indices[val_valid_mask]
feature_data_val = feature_data_val[val_valid_mask]
val_df_filtered = val_df.iloc[val_valid_mask]

test_valid_mask = np.isin(test_indices, valid_indices)
test_valid_indices = test_indices[test_valid_mask]
feature_data_test = feature_data_test[test_valid_mask]
test_df_filtered = test_df.iloc[test_valid_mask]


"""
Action Label Generation
Generates action labels (Long, Short, No Trade) for training, validation, and test sets using profit data. Applies the `generate_action_labels` function to assign labels based on a reward function and profit thresholds. Uses filtered indices to ensure valid profit calculations. Logs the process and distribution of labels for each set.
"""

# Generate action labels
print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Generating action labels...")
train_labels = generate_action_labels(long_profits[train_valid_indices], short_profits[train_valid_indices], train_valid_indices, len(train_df_filtered), MAX_PENALTY, K, is_test_set=False)
val_labels = generate_action_labels(long_profits[val_valid_indices], short_profits[val_valid_indices], val_valid_indices, len(val_df_filtered), MAX_PENALTY, K, is_test_set=False)
test_labels = generate_action_labels(long_profits[test_valid_indices], short_profits[test_valid_indices], test_valid_indices, len(test_df_filtered), MAX_PENALTY, K, is_test_set=True)
#     target_long_trades = int(total_bars * 0.3)
#     target_short_trades = int(total_bars * 0.3)


# print (available_long , available_short , total_bars , max_trade_ratio)


"""
SMOTE Application for Class Balancing
Applies SMOTE (Synthetic Minority Oversampling Technique) to balance the training dataset across three classes: Long (0), Short (1), and No Trade (2). SMOTE generates synthetic samples for minority classes to address imbalance, using 3 nearest neighbors (`k_neighbors=3`). Logs the number of samples before and after SMOTE, along with the resulting class distribution. If SMOTE fails, reverts to original data and logs the failure.

Reason for Applying SMOTE: The dataset likely has an imbalanced class distribution, with "No Trade" being the majority class due to selective trade criteria (e.g., high-profit thresholds). SMOTE ensures better representation of Long and Short classes, improving model performance on minority classes.

Logic: SMOTE creates synthetic samples by interpolating between existing minority class samples, preserving data characteristics. It is applied to all three classes if any are underrepresented, aiming for equal class counts in `train_labels_balanced`. The `random_state=42` ensures reproducibility, and `k_neighbors=3` controls the interpolation range.

Reason for Multi-Class SMOTE: SMOTE works for three classes by treating each minority class independently, generating synthetic samples to balance the dataset. The code validates this by logging the resulting class counts (`Long`, `Short`, `No Trade`), ensuring they are roughly equal post-SMOTE.
"""

print("Before Smote: ",np.unique(train_labels,return_counts = True))
# Apply SMOTE for better class balancing
print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Applying SMOTE for class balancing...")
smote = SMOTE(random_state=42, k_neighbors=3)
try:# remove smote
    feature_data_train_balanced, train_labels_balanced = smote.fit_resample(feature_data_train, train_labels)
    
    print(f"SMOTE Applied: {len(feature_data_train)} -> {len(feature_data_train_balanced)} samples\n")
    print(f"Balanced Distribution: Long={np.sum(train_labels_balanced==0)}, Short={np.sum(train_labels_balanced==1)}, No Trade={np.sum(train_labels_balanced==2)}\n")
except:
    feature_data_train_balanced = feature_data_train
    train_labels_balanced = train_labels
    print("SMOTE failed, using original data\n")


# """
# Hyperparameter Tuning
# Performs grid search to optimize XGBoost model hyperparameters using balanced training data. 
# Tests combinations of `max_depth`, `learning_rate`, `n_estimators`, and other parameters to maximize balanced accuracy on the validation set. 
# Logs each combination's performance and the best parameters found. Returns the optimal hyperparameters for model training.
# """

# # Hyperparameter tuning
# best_params = hyperparameter_tuning(feature_data_train_balanced, train_labels_balanced, feature_data_val, val_labels)




best_params = {'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 1000, 'min_child_weight': 3, 'subsample': 0.9, 'colsample_bytree': 0.8}

"""
Ensemble Model Training
Trains XGBoost and Random Forest models on balanced training data for multi-class classification (Long, Short, No Trade). 
XGBoost uses optimized hyperparameters with `multi:softprob` objective, 
while Random Forest uses fixed parameters with balanced class weights. 
Saves both models to specified file paths for later use. Logs the training process and model file locations.
"""

# Train ensemble models
print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Training ensemble models...")

# XGBoost model
xgb_model = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    **best_params
)
xgb_model.fit(
    feature_data_train_balanced, train_labels_balanced,
    eval_set=[(feature_data_val, val_labels)],
    verbose=False
)

# Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,#problem discuss with leafs, depth
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(feature_data_train_balanced, train_labels_balanced)

# Save models and feature information
joblib.dump(xgb_model, DIRECTORIES['XGB_TRAINED_MODEL_FILE'])
joblib.dump(rf_model, DIRECTORIES['RF_TRAINED_MODEL_FILE'])


"""
Model Evaluation
Evaluates the ensemble model (XGBoost and Random Forest) on the test dataset using price features. Computes predictions, confusion matrix, and trading performance metrics like total profit and trade counts. Logs detailed evaluation results, including accuracy and per-class metrics, to the documentation file. Returns a dictionary with key performance metrics for further analysis.
"""

# EVALUATE MODEL
print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Evaluating model...")

# Evaluate model
print("Model Evaluation (With Price Features)\n")
# results = evaluate_model(xgb_model, rf_model, feature_data_test, test_labels, test_valid_indices, 
#               long_profits, short_profits, test_df_filtered, "Model")

results = evaluate_model(xgb_model, rf_model, feature_data_test, test_labels, test_valid_indices, test_df_filtered, "Model", long_profits, short_profits, MAX_PENALTY, K)

# ===== EXPORT TRADING DATA =====
print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Starting data export...")

# Export trading data
export_results = export_trading_data(df, feature_data_test, test_labels, test_valid_indices, test_df_filtered, 
                                   xgb_model, rf_model, DIRECTORIES, SYMBOL, SUFFIX, VERSION)

print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: All exports completed successfully!")
print("Files created:")
print(f"  - Raw OHLC data: {export_results['ohlc_filename']}")
if export_results['trade_filename']:
    print(f"  - Trade signals: {export_results['trade_filename']}")



import numpy as np
import xgboost as xgb
from datetime import datetime

def train_xgboost_model(X_train, y_train, X_val, y_val):
    """Train XGBoost model with updated parameters"""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Training XGBoost model...")
    
    # Calculate class weights
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    class_weights = {
        0: total_samples / (3 * class_counts[0]),  # Long
        1: total_samples / (3 * class_counts[1]),  # Short
        2: total_samples / (3 * class_counts[2])   # No Trade
    }
    
    # Updated XGBoost parameters
    xgb_params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'max_depth': 6,  # Increased depth
        'learning_rate': 0.05,  # Increased learning rate
        'n_estimators': 1000,  # More trees
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1,
        'scale_pos_weight': [class_weights[0], class_weights[1], class_weights[2]],  # Class weights
        'tree_method': 'hist',
        'random_state': 42
    }
    
    # Create and train model
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=100
    )
    
    return model
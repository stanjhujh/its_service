import numpy as np
from xgboost import XGBClassifier
from datetime import datetime

def hyperparameter_tuning(X_train, y_train, X_val, y_val):
    
    """Perform hyperparameter tuning using grid search"""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Starting hyperparameter tuning...")
    
    # Define parameter grid
    param_grid = {
        'max_depth': [1,2,4, 5, 6,7,8],
        'learning_rate': [0.01,0.03, 0.05, 0.1,0.3,0.5],
        'n_estimators': [150,500, 800, 1000,1500],
        'min_child_weight': [1, 3, 5,7,9],
        'subsample': [0.5,0.6,0.7,0.8, 0.9],
        'colsample_bytree': [0.5,0.6,0.7,0.8, 0.9]
    }
    
    best_score = 0
    best_params = None
    
    # Calculate class weights
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    class_weights = {
        0: total_samples / (3 * class_counts[0]),  # Long
        1: total_samples / (3 * class_counts[1]),  # Short
        2: total_samples / (3 * class_counts[2])   # No Trade
    }
    sample_weights = np.array([class_weights[label] for label in y_train])
    
    # Grid search with reduced combinations for speed
    # Existing combinations preserved
    param_combinations = [
        {'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 800, 'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.8},
        {'max_depth': 4, 'learning_rate': 0.1, 'n_estimators': 500, 'min_child_weight': 3, 'subsample': 0.9, 'colsample_bytree': 0.9},
        {'max_depth': 6, 'learning_rate': 0.03, 'n_estimators': 1000, 'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.8},
        {'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 1000, 'min_child_weight': 3, 'subsample': 0.9, 'colsample_bytree': 0.8},
        {'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 800, 'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.9},

        # New unique combinations added below:
        {'max_depth': 7, 'learning_rate': 0.3, 'n_estimators': 1500, 'min_child_weight': 5, 'subsample': 0.7, 'colsample_bytree': 0.7},
        {'max_depth': 2, 'learning_rate': 0.01, 'n_estimators': 500, 'min_child_weight': 9, 'subsample': 0.6, 'colsample_bytree': 0.6},
        {'max_depth': 1, 'learning_rate': 0.5, 'n_estimators': 800, 'min_child_weight': 7, 'subsample': 0.5, 'colsample_bytree': 0.5},
        {'max_depth': 8, 'learning_rate': 0.1, 'n_estimators': 150, 'min_child_weight': 3, 'subsample': 0.9, 'colsample_bytree': 0.6},
        {'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 1000, 'min_child_weight': 5, 'subsample': 0.7, 'colsample_bytree': 0.9}
    ]

    
    for i, params in enumerate(param_combinations):
        print(f"Testing combination {i+1}/{len(param_combinations)}: {params}")
        
        # Create model with current parameters
        model = XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            **params
        )
        
        # Train model
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Evaluate on validation set
        val_probs = model.predict_proba(X_val)
        
        # Custom prediction logic
        val_predictions = np.full(len(X_val), 2)
        for j in range(len(val_probs)):
            long_prob, short_prob, no_trade_prob = val_probs[j]
            if long_prob > 0.3 and long_prob > short_prob and long_prob > no_trade_prob:
                val_predictions[j] = 0
            elif short_prob > 0.3 and short_prob > long_prob and short_prob > no_trade_prob:
                val_predictions[j] = 1
        
        # Calculate accuracy
        accuracy = np.mean(val_predictions == y_val)
        
        # Calculate balanced accuracy (giving equal weight to each class)
        class_accuracies = []
        for class_label in [0, 1, 2]:
            class_mask = y_val == class_label
            if np.sum(class_mask) > 0:
                class_acc = np.mean(val_predictions[class_mask] == class_label)
                class_accuracies.append(class_acc)
        
        balanced_accuracy = np.mean(class_accuracies) if class_accuracies else 0
        
        if balanced_accuracy > best_score:
            best_score = balanced_accuracy
            best_params = params
    
    return best_params
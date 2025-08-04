from datetime import datetime

def select_important_features(df, n_features=76):
    """Select features directly from the static SELECTED_FEATURES list to ensure exact consistency."""
    
    # Use the static SELECTED_FEATURES list directly
    from __main__ import SELECTED_FEATURES
    
    found_features = []
    missing_features = []
    
    # Debug: Print the actual SELECTED_FEATURES list
    print(f"SELECTED_FEATURES length: {len(SELECTED_FEATURES)}")
    
    # Check which features from SELECTED_FEATURES exist in the dataframe
    for feat in SELECTED_FEATURES:
        if feat in df.columns:
            found_features.append(feat)
        else:
            missing_features.append(feat)
    
    print(f"Found {len(found_features)} features for model.")
    print(f"Missing features: {missing_features}")
    
    
    # Ensure we don't exceed the number of found features
    actual_features = found_features[:min(n_features, len(found_features))]
    
    print(f"Final selected features count: {len(actual_features)}\n")
    
    # Verify we have exactly the expected number of features
    if len(actual_features) != len(SELECTED_FEATURES):
        print(f"WARNING: Feature count mismatch! Expected {len(SELECTED_FEATURES)}, got {len(actual_features)}\n")
        if missing_features:
            print(f"Missing features that need to be added to dataset: {missing_features}\n")
    
    return actual_features
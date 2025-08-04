import numpy as np
import pandas as pd
from datetime import datetime
import os
from reward_function import reward

def evaluate_model(xgb_model, rf_model, feature_data_test, test_labels, test_valid_indices, test_df, model_name, long_profits, short_profits, MAX_PENALTY, K):
    """Evaluate a model and log results with confusion matrices and detailed metrics"""
    
    # Get predictions from both models
    xgb_probs = xgb_model.predict_proba(feature_data_test)
    rf_probs = rf_model.predict_proba(feature_data_test)
    
    # Ensemble averaging with weights (XGB gets higher weight due to better performance)
    ensemble_probs = 0.7 * xgb_probs + 0.3 * rf_probs
    
    # Adaptive thresholding based on profit percentiles
    predicted_actions = np.full(len(feature_data_test), 2)  # Default to No Trade
    
    
    # Apply thresholds with profit considerations
    for i in range(len(ensemble_probs)):
        long_prob, short_prob, no_trade_prob = ensemble_probs[i]
        
        # Check if meets threshold and profit criteria
        test_idx = test_valid_indices[i] if i < len(test_valid_indices) else 0
        
        if (long_prob > short_prob and long_prob > no_trade_prob):  
            predicted_actions[i] = 0  # Long
        elif (short_prob >= long_prob and short_prob > no_trade_prob):
            predicted_actions[i] = 1  # Short
        # Otherwise remains No Trade (2)
    
    # ===== SAVE DETAILED RESULTS WITH DATES AND SIGNALS =====
    # Create detailed results DataFrame
    results_data = []
    signal_names = {0: 'Long', 1: 'Short', 2: 'No Trade'}
    
    # Get feature names if feature_data_test is a DataFrame, otherwise use generic names
    if hasattr(feature_data_test, 'columns'):
        feature_names = feature_data_test.columns.tolist()
    else:
        # If it's a numpy array, create generic feature names
        feature_names = [f'Feature_{i}' for i in range(feature_data_test.shape[1])]
    
    for i in range(len(feature_data_test)):
        test_idx = test_valid_indices[i] if i < len(test_valid_indices) else 0
        
        # Get date from test_df if available
        if 'Date' in test_df.columns:
            date = test_df.iloc[test_idx]['Date'] if test_idx < len(test_df) else None
        elif 'datetime' in test_df.columns:
            date = test_df.iloc[test_idx]['datetime'] if test_idx < len(test_df) else None
        else:
            date = None
            
        # Get probabilities
        long_prob, short_prob, no_trade_prob = ensemble_probs[i]
        
        # Get actual and predicted signals
        actual_signal = signal_names.get(test_labels[i], 'Unknown')
        predicted_signal = signal_names.get(predicted_actions[i], 'Unknown')
        
        # Get profit information
        long_profit = long_profits[test_idx] if test_idx < len(long_profits) else 0
        short_profit = short_profits[test_idx] if test_idx < len(short_profits) else 0
        
        # Determine which profit to use based on actual signal
        actual_profit = 0
        if test_labels[i] == 0:  # Long
            actual_profit = long_profit
        elif test_labels[i] == 1:  # Short
            actual_profit = short_profit
            
        # Calculate reward
        actual_reward = reward(test_labels[i], long_profit, short_profit, MAX_PENALTY, K)
        
        # Create base record with all existing fields
        record = {
            'Date': date,
            'Index': test_idx,
            'Actual_Signal': actual_signal,
            'Predicted_Signal': predicted_signal,
            'Long_Probability': long_prob,
            'Short_Probability': short_prob,
            'NoTrade_Probability': no_trade_prob,
            'Long_Profit': long_profit,
            'Short_Profit': short_profit,
            'Actual_Profit': actual_profit,
            'Actual_Reward': actual_reward,
            'Correct_Prediction': test_labels[i] == predicted_actions[i],
            'Model_Name': model_name
        }
        
        # Add input features to the record
        if hasattr(feature_data_test, 'iloc'):
            # If it's a DataFrame, get features by column names
            for feature_name in feature_names:
                record[feature_name] = feature_data_test.iloc[i][feature_name]
        else:
            # If it's a numpy array, get features by index
            for j, feature_name in enumerate(feature_names):
                record[feature_name] = feature_data_test[i, j]
        
        results_data.append(record)
    
    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(results_data)
    
    # Create output directory if it doesn't exist
    output_dir = 'evaluation_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/{model_name}_evaluation_results_{timestamp}.csv"
    
    # Save to CSV
    results_df.to_csv(filename, index=False)
    print(f"\n### Detailed Results Saved\n")
    print(f"Results saved to: {filename}\n")
    print(f"Total records saved: {len(results_df)}\n")
    
    # Print summary of saved data
    print(f"Signal Distribution in Saved Data:\n")
    print(f"Actual Signals: {results_df['Actual_Signal'].value_counts().to_dict()}\n")
    print(f"Predicted Signals: {results_df['Predicted_Signal'].value_counts().to_dict()}\n")
    print(f"Correct Predictions: {results_df['Correct_Prediction'].sum()}/{len(results_df)} ({results_df['Correct_Prediction'].mean()*100:.2f}%)\n")
    
    # ===== CONFUSION MATRIX ANALYSIS =====
    print(f"## {model_name} Detailed Evaluation\n")
    
    # Calculate confusion matrix
    confusion_mat = np.zeros((3, 3), dtype=int)
    for true, pred in zip(test_labels, predicted_actions):
        confusion_mat[true][pred] += 1
    
    # Print confusion matrix
    class_names = ['Long (0)', 'Short (1)', 'No Trade (2)']
    print("### Confusion Matrix:\n")
    print("Predicted →\n")
    print("Actual ↓    Long    Short    No Trade\n")
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<10} {confusion_mat[i][0]:^8} {confusion_mat[i][1]:^8} {confusion_mat[i][2]:^10}\n")
    
    # Calculate detailed statistics
    print("\n### Detailed Statistics:\n")
    for true_class in range(3):
        total_actual = np.sum(test_labels == true_class)
        if total_actual == 0:
            continue
            
        correct = confusion_mat[true_class][true_class]
        accuracy = correct / total_actual if total_actual > 0 else 0
        
        print(f"\nActual {class_names[true_class]}:\n")
        print(f"Total Cases: {total_actual}\n")
        print(f"Correct Predictions: {correct} ({accuracy*100:.2f}%)\n")
        
        # Show misclassifications
        for pred_class in range(3):
            if pred_class != true_class:
                misclassified = confusion_mat[true_class][pred_class]
                if misclassified > 0:
                    print(f"Misclassified as {class_names[pred_class]}: {misclassified} ({misclassified/total_actual*100:.2f}%)\n")
    
    # Overall accuracy
    total_correct = np.sum(np.diag(confusion_mat))
    total_cases = np.sum(confusion_mat)
    overall_accuracy = total_correct / total_cases if total_cases > 0 else 0
    print(f"\nOverall Statistics:\n")
    print(f"Total Test Cases: {total_cases}\n")
    print(f"Total Correct Predictions: {total_correct} ({overall_accuracy*100:.2f}%)\n")
    print(f"Total Misclassifications: {total_cases - total_correct} ({(1-overall_accuracy)*100:.2f}%)\n")
    
    # ===== TRADING PERFORMANCE METRICS =====
    # Evaluation metrics
    actions = predicted_actions
    long_mask = (actions == 0) & test_df['InWindow'].values
    short_mask = (actions == 1) & test_df['InWindow'].values
    long_trades = np.sum(long_mask)
    short_trades = np.sum(short_mask)
    long_profits_list = [long_profits[test_valid_indices[i]] for i in range(len(actions)) if actions[i] == 0]
    short_profits_list = [short_profits[test_valid_indices[i]] for i in range(len(actions)) if actions[i] == 1]
    total_long_profit = sum(long_profits_list) if long_profits_list else 0
    total_short_profit = sum(short_profits_list) if short_profits_list else 0
    avg_long_profit = np.mean(long_profits_list) if long_profits_list else 0
    avg_short_profit = np.mean(short_profits_list) if short_profits_list else 0
    long_rewards = [reward(0, long_profits[test_valid_indices[i]], short_profits[test_valid_indices[i]], MAX_PENALTY, K) for i in range(len(actions)) if actions[i] == 0]
    short_rewards = [reward(1, long_profits[test_valid_indices[i]], short_profits[test_valid_indices[i]], MAX_PENALTY, K) for i in range(len(actions)) if actions[i] == 1]
    total_long_reward = sum(long_rewards) if long_rewards else 0
    total_short_reward = sum(short_rewards) if short_rewards else 0
    avg_long_reward = np.mean(long_rewards) if long_rewards else 0
    avg_short_reward = np.mean(short_rewards) if short_rewards else 0
    
    print(f"### {model_name} Trading Performance\n")
    print(f"#### Long Trades\n")
    print(f"Total Base Profit: ${total_long_profit:.2f}\n")
    print(f"Average Base Profit: ${avg_long_profit:.2f}, {long_trades} trades\n")
    print(f"Total Reward: {total_long_reward:.2f}\n")
    print(f"Average Reward: {avg_long_reward:.2f}\n")
    print(f"Total Trades: {long_trades} trades, {long_trades/len(test_df)*100:.2f}% of test set bars\n")
    profitable_long = sum(1 for p in long_profits_list if p > 0)
    print(f"Profitable Long Trades: {profitable_long}/{long_trades} ({profitable_long/max(long_trades,1)*100:.1f}%)\n")
    
    print(f"#### Short Trades\n")
    print(f"Total Base Profit: ${total_short_profit:.2f}\n")
    print(f"Average Base Profit: ${avg_short_profit:.2f}, {short_trades} trades\n")
    print(f"Total Reward: {total_short_reward:.2f}\n")
    print(f"Average Reward: {avg_short_reward:.2f}\n")
    print(f"Total Trades: {short_trades} trades, {short_trades/len(test_df)*100:.2f}% of test set bars\n")
    profitable_short = sum(1 for p in short_profits_list if p > 0)
    print(f"Profitable Short Trades: {profitable_short}/{short_trades} ({profitable_short/max(short_trades,1)*100:.1f}%)\n")
    
    print(f"#### Overall Trading Statistics\n")
    print(f"Action Distribution: {np.bincount(actions, minlength=3)/len(actions)*100}\n")
    print(f"Total Profit: ${total_long_profit + total_short_profit:.2f}\n")
    print(f"Overall Accuracy: {np.mean(predicted_actions == test_labels)*100:.2f}%\n")
    
    # ===== ADDITIONAL METRICS =====
    # Calculate precision, recall, F1-score for each class
    print(f"#### Precision, Recall, F1-Score by Class\n")
    for class_label in range(3):
        class_name = class_names[class_label]
        tp = confusion_mat[class_label][class_label]  # True positives
        fp = np.sum(confusion_mat[:, class_label]) - tp  # False positives
        fn = np.sum(confusion_mat[class_label, :]) - tp  # False negatives
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{class_name}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1_score:.3f}\n")
    
    #Return key metrics for comparison
    return {
        'model_name': model_name,
        'overall_accuracy': overall_accuracy,
        'total_profit': total_long_profit + total_short_profit,
        'long_trades': long_trades,
        'short_trades': short_trades,
        'avg_long_profit': avg_long_profit,
        'avg_short_profit': avg_short_profit,
        'profitable_long_rate': profitable_long/max(long_trades,1)*100,
        'profitable_short_rate': profitable_short/max(short_trades,1)*100,
        'confusion_matrix': confusion_mat,
        'results_file': filename,
        'results_dataframe': results_df
    }


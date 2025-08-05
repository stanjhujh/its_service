import numpy as np
import pandas as pd
from datetime import datetime
from reward_function import reward

# Generate action labels with improved strategy
def generate_action_labels(long_profits, short_profits, indices, total_bars, MAX_PENALTY, K, is_test_set=False):
     print(f"Total bars: {total_bars}")
    
    labels = np.full(total_bars, 2)
    
    profit_df = pd.DataFrame({
        'index': indices,
        'long_profit': long_profits,
        'short_profit': short_profits,
        'long_reward': [reward(0, lp, sp) for lp, sp in zip(long_profits, short_profits)],
        'short_reward': [reward(1, lp, sp) for lp, sp in zip(long_profits, short_profits)]
    })
    
    # Calculate additional metrics for better selection
    profit_df['long_profit_ratio'] = np.where(profit_df['short_profit'] != 0, 
                                            profit_df['long_profit'] / np.abs(profit_df['short_profit']), 
                                            profit_df['long_profit'])
    profit_df['short_profit_ratio'] = np.where(profit_df['long_profit'] != 0, 
                                             profit_df['short_profit'] / np.abs(profit_df['long_profit']), 
                                             profit_df['short_profit'])
    
    # Enhanced selection criteria
    
    # Step 1: Select high-quality trades with multiple criteria
    long_candidates = profit_df[
        (profit_df['long_profit'] > min_profit_threshold) & 
        (profit_df['long_reward'] > 0) &
        (profit_df['long_profit'] > profit_df['short_profit'])
    ].copy()
    
    short_candidates = profit_df[
        (profit_df['short_profit'] > min_profit_threshold) & 
        (profit_df['short_reward'] > 0) &
        (profit_df['short_profit'] > profit_df['long_profit'])
    ].copy()
    
    # Calculate composite scores for ranking
    long_candidates['composite_score'] = (
        long_candidates['long_profit'] * 0.4 +
        long_candidates['long_reward'] * 0.3 +
        long_candidates['long_profit_ratio'] * 0.3
    )
    
    short_candidates['composite_score'] = (
        short_candidates['short_profit'] * 0.4 +
        short_candidates['short_reward'] * 0.3 +
        short_candidates['short_profit_ratio'] * 0.3
    )
    
    available_long = len(long_candidates)
    available_short = len(short_candidates)

    print(f"available_long: {available_long}")
    print(f"available_short: {available_short}")

    # Calculate current trade ratios
    long_ratio_data = available_long / total_bars if total_bars > 0 else 0
    short_ratio_data = available_short / total_bars if total_bars > 0 else 0
    total_trade_ratio = (available_long + available_short) / total_bars if total_bars > 0 else 0

    print(f"long_ratio: {long_ratio}")
    print(f"short_ratio: {short_ratio}")

    print(f"total_trade_ratio: {total_trade_ratio}")
    
    
    # Select top trades based on composite score
    if len(long_candidates) > 0:
        final_long_indices = long_candidates.nlargest(target_long_trades highest 5% profit, 'composite_score').index
        labels[final_long_indices] = 0  # Long
    
    if len(short_candidates) > 0:
        final_short_indices = short_candidates.nlargest(target_short_trades highest 5% profit, 'composite_score').index
        labels[final_short_indices] = 1  # Short

    return labels
import numpy as np
import pandas as pd
from datetime import datetime
from reward_function import reward

# Generate action labels with improved strategy
def generate_action_labels(long_profits, short_profits, indices, total_bars, MAX_PENALTY, K, is_test_set=False):
    """Generate action labels with improved profit-based selection and balanced distribution"""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Generating improved action labels...")
    
    # Initialize labels array with No Trade (2)
    labels = np.full(len(long_profits), 2)
    
    # Create DataFrame for ranking with enhanced metrics
    profit_df = pd.DataFrame({
        'long_profit': long_profits,
        'short_profit': short_profits,
        'long_reward': [reward(0, lp, sp, MAX_PENALTY, K) for lp, sp in zip(long_profits, short_profits)],
        'short_reward': [reward(1, lp, sp, MAX_PENALTY, K) for lp, sp in zip(long_profits, short_profits)]
    })
    
    # Calculate additional metrics for better selection
    profit_df['long_profit_ratio'] = np.where(profit_df['short_profit'] != 0, 
                                            profit_df['long_profit'] / np.abs(profit_df['short_profit']), 
                                            profit_df['long_profit'])
    profit_df['short_profit_ratio'] = np.where(profit_df['long_profit'] != 0, 
                                             profit_df['short_profit'] / np.abs(profit_df['long_profit']), 
                                             profit_df['short_profit'])
    
    # Enhanced selection criteria
    # Use higher thresholds for better quality trades
    min_profit_threshold = 35   # Increased from 30
    
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
    
    # Determine target distribution based on available profitable trades
    total_bars = len(profit_df)
    available_long = len(long_candidates)
    available_short = len(short_candidates)
    
    # Adaptive target distribution based on available profitable trades, #### have to review that 
    max_trade_ratio = 0.75  # Maximum 20% trades total
    print (available_long , available_short , total_bars , max_trade_ratio)
    if available_long + available_short < total_bars * max_trade_ratio:
        # Use all available profitable trades
        print(f'entry condition 1')
        
        target_long_trades = int(int(available_long) * 1)
        target_short_trades = int(int(available_short) * 0.7)
        
        
#         target_long_trades = min(available_long, int(total_bars * 1))
#         target_short_trades = min(available_short, int(total_bars * 0.5))
    else:
        print(f'entry condition 2')
        # Use standard 12.5% each
        target_long_trades = int(total_bars * 1)
        target_short_trades = int(total_bars * 1)

    print(target_long_trades, target_short_trades)
    
    print(type(long_candidates),long_candidates.shape)
    print(type(short_candidates),short_candidates.shape)
    
    print(target_long_trades)
    print(target_short_trades)
    
    # Select top trades based on composite score
    if len(long_candidates) > 0:
        final_long_indices = long_candidates.nlargest(target_long_trades, 'composite_score').index
        labels[final_long_indices] = 0  # Long
    
    if len(short_candidates) > 0:
        final_short_indices = short_candidates.nlargest(target_short_trades, 'composite_score').index
        labels[final_short_indices] = 1  # Short

    
    return labels
import numpy as np

# Reward function
def reward(action, long_profit, short_profit, MAX_PENALTY, K):
    if action == 0:  # Long
        if long_profit > 30:
            return long_profit
        elif 10 < long_profit <= 30:
            return -MAX_PENALTY * np.exp(-K * (long_profit - 10))
        else:
            return -MAX_PENALTY
    elif action == 1:  # Short
        if short_profit > 30:
            return short_profit
        elif 10 < short_profit <= 30:
            return -MAX_PENALTY * np.exp(-K * (short_profit - 10))
        else:
            return -MAX_PENALTY
    else:  # No Trade
        return 0
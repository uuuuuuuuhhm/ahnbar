Project Module: Bankroll Management & Kelly Criterion
Goal: Implement a betting stake calculator based on model edge.
Logic Requirements:
Input: Accept model_win_prob (0.0 to 1.0) and market_decimal_odds.
Calculate Edge: Determine if model_win_prob > (1 / market_decimal_odds). If no edge exists, stake = 0.
Kelly Formula:
b = market_decimal_odds - 1
p = model_win_prob
q = 1 - p
fraction = (p * b - q) / b
Fractional Kelly: Implement a 'Safety Multiplier' (default to 0.25 for "Quarter Kelly"). This reduces volatility by only betting 1/4 of the suggested amount.
Output: Return the suggested stake as a percentage of the total current_bankroll.
Safety Constraint: Never suggest a stake higher than 5% of the total bankroll, regardless of how high the model's confidence is.
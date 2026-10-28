"""Step 4: the cheat-sheet thresholds using ONLY inning, strikes and challenges left.

Balls, runners, outs and score are ignored: each box is
    average cost / (average gain + average cost)
over every wrong call in that inning / strikes / challenges-left group.
Needs decision_rows.csv (from step 3). Writes threshold_simple.csv
"""
import pandas as pd

D = pd.read_csv("decision_rows.csv")
D["strikes"] = D["strikes"].astype(int)
D["inn_bin"] = D["inn_bin"].astype(str)
keys = ["inn_bin", "strikes", "remaining"]

S = D.groupby(keys).agg(n=("gain", "size"), gain=("gain", "mean"), cost=("cost", "mean"))
S["p_simple"] = S["cost"] / (S["gain"] + S["cost"])
print(S.round(3).to_string())
S.to_csv("threshold_simple.csv")

# how much does ignoring balls and runners change the answer?
D = D.merge(S["p_simple"].reset_index(), on=keys)
print("average gap between simple and full threshold:",
      round((D["p_simple"] - D["p_needed"]).abs().mean(), 3))
for c in [0.1, 0.2, 0.3, 0.5]:
    flip = ((D["p_simple"] <= c) != (D["p_needed"] <= c)).mean()
    print(f"at {int(c*100)}% confidence, the simple rule disagrees with the full rule on {flip:.0%} of wrong calls")

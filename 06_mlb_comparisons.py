"""Step 6: MLB comparison checks.

(a) Is a hitter's swing-vs-take score near the zone edge a stable trait,
    and does it line up with challenge skill?  (In 2026 data: no.)
(b) Which direction are wrong called strikes (side / high / low)?

Needs pitches_2026.csv, takes_labeled_2026.csv and player_discrimination.csv.
"""
import numpy as np
import pandas as pd
from abs_utils import edge_dist

# ---------------- (a) swing score ----------------
cols = ["game_type", "description", "plate_x", "plate_z", "sz_top", "sz_bot", "batter", "game_pk"]
sc = pd.read_csv("pitches_2026.csv", usecols=cols)
sc = sc[sc["game_type"] == "R"].copy()

take_d = ["called_strike", "ball", "blocked_ball"]
swing_d = ["swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play"]
sc["swing"] = sc["description"].isin(swing_d)
sc = sc[sc["swing"] | sc["description"].isin(take_d)].copy()

f = lambda c: sc[c].to_numpy(dtype="float64", na_value=np.nan)
sc["dist_in"] = edge_dist(f("plate_x"), f("plate_z"), f("sz_top"), f("sz_bot"))
sc = sc.dropna(subset=["dist_in"])

bd = sc[sc["dist_in"].abs() <= 2.0].copy()
bd["zone"] = np.where(bd["dist_in"] > 0, "inside", "outside")
mlb = bd.groupby(["batter", "zone"])["swing"].agg(["count", "mean"]).unstack("zone")
mlb.columns = [f"{a}_{z}" for a, z in mlb.columns]
mlb = mlb[(mlb["count_inside"] >= 40) & (mlb["count_outside"] >= 40)].copy()
mlb["swing_disc"] = mlb["mean_inside"] - mlb["mean_outside"]
print(len(mlb), "MLB hitters | average swing_disc:", round(mlb["swing_disc"].mean(), 3))

bd["half"] = np.where(bd["game_pk"] % 2 == 0, "A", "B")
hh = bd.groupby(["batter", "half", "zone"])["swing"].mean().unstack(["half", "zone"])
hh = hh.loc[hh.index.isin(mlb.index)].dropna()
a = hh[("A", "inside")] - hh[("A", "outside")]
b2 = hh[("B", "inside")] - hh[("B", "outside")]
print("swing_disc split-half correlation:", round(a.corr(b2), 2), "|", len(hh), "hitters")

dj = pd.read_csv("player_discrimination.csv")
dj = dj[dj["side"] == "batter"].copy()
dj["batter"] = dj["player"].astype("int64")
mm = mlb.reset_index().merge(dj[["batter", "wrong_surplus", "right_surplus", "discrimination"]], on="batter")
print(len(mm), "hitters with both scores")
for c in ["wrong_surplus", "right_surplus", "discrimination"]:
    print(c, round(mm["swing_disc"].corr(mm[c]), 2))

# ---------------- (b) direction of wrong called strikes ----------------
def direction(x_ft, z_ft, top_ft, bot_ft):
    ex = np.column_stack([np.abs(x_ft) * 12 - 8.5,      # off the side
                          (z_ft - top_ft) * 12,          # above the top
                          (bot_ft - z_ft) * 12])         # below the bottom
    return np.array(["side", "high", "low"])[np.argmax(ex, axis=1)]


Lb = pd.read_csv("takes_labeled_2026.csv")
orig_strike = (Lb["description"] == "called_strike") ^ (Lb["record"] == "challenged_overturned")
Mw = Lb[orig_strike & (Lb["dist_in"] < 0)].copy()      # wrong ORIGINAL called strikes, incl. overturned
fm = lambda c: Mw[c].to_numpy(dtype="float64", na_value=np.nan)
Mw["direction"] = direction(fm("plate_x"), fm("plate_z"), fm("sz_top"), fm("sz_bot"))
Mw["margin"] = -Mw["dist_in"]
print("MLB | wrong original called strikes:", len(Mw))
print("  all:", Mw["direction"].value_counts(normalize=True).round(3).to_dict())
big = Mw[Mw["margin"] >= 3]
print("  3+ inches off:", len(big), big["direction"].value_counts(normalize=True).round(3).to_dict())
print("  share challenged and overturned:", round((Mw["record"] == "challenged_overturned").mean(), 3))

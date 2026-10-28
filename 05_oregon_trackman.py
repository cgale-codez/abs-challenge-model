"""Step 5: score Oregon takes against the ABS zone using TrackMan files.

Put your TrackMan CSVs in a folder called  trackman/  (subfolders are fine).
First run: writes oregon_heights_template.csv -> fill in HeightInches for each hitter, save, run again.
Writes oregon_zone_awareness.csv and oregon_wrong_calls_for_film.csv

PRIVATE DATA: do not commit trackman/ or the oregon_* files (see .gitignore).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from abs_utils import edge_dist, zone_from_height

FOLDER = "trackman"
BAND = 2.0   # inches from the zone edge; start wide while you have few games, tighten later
TEAM = "ORE_DUC"

# ---- load every TrackMan file, drop duplicate pitches ----
paths = glob.glob(os.path.join(FOLDER, "**", "*.csv"), recursive=True)
frames, skipped = [], []
for p in paths:
    cols = pd.read_csv(p, nrows=0).columns
    if "PitchUID" in cols and "PitchCall" in cols:
        frames.append(pd.read_csv(p))
    else:
        skipped.append(os.path.basename(p))
tm = pd.concat(frames, ignore_index=True)
before = len(tm)
tm = tm.drop_duplicates("PitchUID")
print(len(frames), "files used |", len(skipped), "skipped")
print(before, "rows ->", len(tm), "after removing duplicate pitches")

# ---- heights file (never overwrite one you already filled in) ----
HEIGHTS = "oregon_heights_template.csv"
if not os.path.exists(HEIGHTS):
    roster = (tm[tm["BatterTeam"] == TEAM][["BatterId", "Batter", "BatterTeam"]]
              .drop_duplicates())
    roster["HeightInches"] = ""
    roster.to_csv(HEIGHTS, index=False)
    print(f"Wrote {HEIGHTS} with {len(roster)} hitters. Fill in HeightInches, save, and run again.")
    sys.exit()

h = pd.read_csv(HEIGHTS)
h["HeightInches"] = pd.to_numeric(h["HeightInches"], errors="coerce")
print("hitters missing a height:", h.loc[h["HeightInches"].isna(), "Batter"].tolist())
h = h.dropna(subset=["HeightInches"])
P = tm.merge(h[["BatterId", "HeightInches"]], on="BatterId", how="inner")

# ---- keep confident tracking, and only takes and swings ----
P = P[P["PitchLocationConfidence"] == "High"]
takes = ["BallCalled", "StrikeCalled"]
swings = ["StrikeSwinging", "FoulBall", "FoulBallNotFieldable", "FoulBallFieldable", "InPlay"]
P = P.copy()
P["kind"] = np.where(P["PitchCall"].isin(takes), "take",
             np.where(P["PitchCall"].isin(swings), "swing", "other"))
P = P[P["kind"] != "other"].copy()

top_ft, bot_ft = zone_from_height(P["HeightInches"])
P["dist_in"] = edge_dist(P["PlateLocSide"].to_numpy(dtype="float64", na_value=np.nan),
                         P["PlateLocHeight"].to_numpy(dtype="float64", na_value=np.nan),
                         top_ft, bot_ft)
P = P.dropna(subset=["dist_in"])

# ---- check 1: do the umpire's calls agree with the ABS zone? ----
T = P[P["kind"] == "take"].copy()
T["agree"] = (T["dist_in"] > 0) == (T["PitchCall"] == "StrikeCalled")
print("overall agreement:", round(T["agree"].mean(), 3), "| takes:", len(T))
print(T.groupby(pd.cut(T["dist_in"], [-30, -3, -2, -1, 0, 1, 2, 3, 30]),
                observed=True)["agree"].agg(["count", "mean"]).round(3))

# ---- check 2: swings inside vs outside the zone near the edge ----
B = P[P["dist_in"].abs() <= BAND].copy()
B["swing"] = (B["kind"] == "swing").astype(int)
B["zone"] = np.where(B["dist_in"] > 0, "inside", "outside")
g = B.groupby(["Batter", "zone"])["swing"].agg(["count", "mean"]).unstack("zone")
g.columns = [f"{a}_{b}" for a, b in g.columns]
g["discrimination"] = g["mean_inside"] - g["mean_outside"]
wcs = (T[(T["PitchCall"] == "StrikeCalled") & (T["dist_in"] < 0)]
       .groupby("Batter").size().rename("called_strikes_outside_zone"))
S = g.join(wcs).fillna({"called_strikes_outside_zone": 0}).sort_values("count_inside", ascending=False)
print(S.round(2).to_string())
S.to_csv("oregon_zone_awareness.csv")

# ---- list of wrong calls to clip in film ----
O = T[T["BatterTeam"] == TEAM].copy()
is_strike = (O["PitchCall"] == "StrikeCalled").to_numpy()
O["wrong"] = (is_strike & (O["dist_in"] < 0)) | (~is_strike & (O["dist_in"] > 0))
O["margin_in"] = np.where(is_strike, -O["dist_in"], O["dist_in"]).round(2)   # > 0 = how wrong
O["against"] = np.where(is_strike, "batter", "pitcher/catcher")
film = (O[O["wrong"]].sort_values("margin_in", ascending=False)
          [["Batter", "Catcher", "Pitcher", "against", "Date", "Time", "Inning", "Top/Bottom",
            "Balls", "Strikes", "PitchCall", "margin_in", "PitchNo", "GameUID"]])
film.to_csv("oregon_wrong_calls_for_film.csv", index=False)
print(len(film), "wrong calls out of", len(O), "takes")
print(film["against"].value_counts())
print(film.head(15).to_string(index=False))

g2 = O.groupby("GameUID").agg(date=("Date", "first"), takes=("wrong", "size"), wrong=("wrong", "sum"))
g2["wrong_rate"] = (g2["wrong"] / g2["takes"]).round(3)
print(g2.sort_values("wrong_rate", ascending=False).to_string())

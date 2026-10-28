"""Step 2: join the three challenge CSVs onto the Statcast takes and compute zone distance.

Needs, in the same folder:
  takes_2026_regular.csv
  taptochallenge_challenges_2026_R.csv
  taptochallenge_missed_2026_R.csv
  taptochallenge_out_of_challenges_2026_R.csv
Writes takes_labeled_2026.csv

The challenge files have no game ID, so each row is matched to a take on
date + teams + inning + count + outs + runners + score + call.
Statcast stores the CORRECTED call for overturned challenges, so for those rows
the original call is flipped before matching.
"""
import numpy as np
import pandas as pd
from abs_utils import edge_dist

takes_r = pd.read_csv("takes_2026_regular.csv")
challenges = pd.read_csv("taptochallenge_challenges_2026_R.csv")
missed = pd.read_csv("taptochallenge_missed_2026_R.csv")
out_of = pd.read_csv("taptochallenge_out_of_challenges_2026_R.csv")

# ---- put the takes in the same shape as the challenge files ----
t = takes_r.copy()
t["Date"] = pd.to_datetime(t["game_date"]).dt.strftime("%Y-%m-%d")
t["Call"] = np.where(t["description"] == "called_strike", "Called Strike", "Ball")

batting = np.where(t["inning_topbot"] == "Top", t["away_team"], t["home_team"])
fielding = np.where(t["inning_topbot"] == "Top", t["home_team"], t["away_team"])
t["Inn"] = t["inning"]
t["Balls"] = t["balls"]
t["Strikes"] = t["strikes"]
t["Outs"] = t["outs_when_up"]

# runners: add 1 for first, 2 for second, 4 for third, then label the result
code = (t["on_1b"].notna().astype(int)
        + 2 * t["on_2b"].notna().astype(int)
        + 4 * t["on_3b"].notna().astype(int))
t["Runners"] = code.map({0: "Empty", 1: "1B", 2: "2B", 4: "3B",
                         3: "1B-2B", 5: "1B-3B", 6: "2B-3B", 7: "Loaded"})

t["bat_team"] = batting
t["fld_team"] = fielding
t["bat_score_i"] = t["bat_score"].astype(int)
t["fld_score_i"] = t["fld_score"].astype(int)
t["stat_call"] = t["Call"]          # the call as Statcast recorded it

# the missed and out-of files have no Call column, so get it from Role
for df in (missed, out_of):
    df["Call"] = np.where(df["Role"].str.lower() == "batter", "Called Strike", "Ball")

key2 = ["Date", "bat_team", "fld_team", "Inn", "Balls", "Strikes", "Outs",
        "Runners", "stat_call", "bat_score_i", "fld_score_i"]


def add_keys(df, flip_overturned=False):
    d = df.copy()
    d["Role"] = d["Role"].str.lower()
    is_bat = d["Role"] == "batter"
    d["bat_team"] = np.where(is_bat, d["Team"], d["Opp"])
    d["fld_team"] = np.where(is_bat, d["Opp"], d["Team"])
    d["bat_score_i"] = np.where(is_bat, d["Team Score"], d["Opp Score"])
    d["fld_score_i"] = np.where(is_bat, d["Opp Score"], d["Team Score"])
    d["stat_call"] = d["Call"]
    if flip_overturned:
        flipped = d["Call"].map({"Ball": "Called Strike", "Called Strike": "Ball"})
        d["stat_call"] = np.where(d["Result"].astype(str) == "True", flipped, d["Call"])
    return d


# only takes whose key is unique can be matched safely
t["take_id"] = np.arange(len(t))
size = t.groupby(key2)["take_id"].transform("size")
unique_takes = t.loc[size == 1, key2 + ["take_id"]]

keep = ["Play ID", "Player", "Role", "Dist (in.)", "xChall%", "xStr%",
        "Challenges Remaining", "Result",
        "Potential Run Expectancy delta if overturned", "Opp Cost RE"]

parts = []
for lab, df, flip in [("challenged", challenges, True),
                      ("missed", missed, False),
                      ("out_of_challenges", out_of, False)]:
    d = add_keys(df, flip)
    mm = d.merge(unique_takes, on=key2, how="inner")
    mm["record"] = lab
    parts.append(mm[[k for k in keep if k in mm.columns] + ["take_id", "record"]])
    print(lab, "matched", len(mm), "of", len(df))

extra = pd.concat(parts, ignore_index=True)
print(extra["take_id"].duplicated().sum(), "takes matched by more than one row")
extra = extra.drop_duplicates("take_id", keep=False)

labeled = t.merge(extra, on="take_id", how="left")
labeled["record"] = labeled["record"].fillna("none")
is_ch = labeled["record"] == "challenged"
labeled["record"] = np.where(is_ch & (labeled["Result"].astype(str) == "True"), "challenged_overturned",
                    np.where(is_ch, "challenged_upheld", labeled["record"]))
print(labeled["record"].value_counts())

# ---- distance from the zone edge, at the front of the plate, plus ball radius ----
labeled["dist_in"] = edge_dist(labeled["plate_x"].to_numpy(dtype="float64", na_value=np.nan),
                               labeled["plate_z"].to_numpy(dtype="float64", na_value=np.nan),
                               labeled["sz_top"].to_numpy(dtype="float64", na_value=np.nan),
                               labeled["sz_bot"].to_numpy(dtype="float64", na_value=np.nan))

known = labeled.dropna(subset=["Dist (in.)"])
err = (known["Dist (in.)"] - known["dist_in"]).abs()
print("max error vs. file Dist:", round(err.max(), 3),
      "| rows off by more than 0.05:", int((err > 0.05).sum()))

is_cs = (labeled["description"] == "called_strike").to_numpy(dtype=bool)
dist = labeled["dist_in"].to_numpy(dtype="float64", na_value=np.nan)
labeled["wrong_call"] = (is_cs & (dist < 0)) | (~is_cs & (dist > 0))
print(pd.crosstab(labeled["record"], labeled["wrong_call"]))

labeled.to_csv("takes_labeled_2026.csv", index=False)

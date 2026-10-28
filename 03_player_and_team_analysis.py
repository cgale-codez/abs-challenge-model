"""Step 3: who challenges well, which teams do, and how sure you need to be.

Needs takes_labeled_2026.csv (from step 2) and the challenge CSVs.
Writes player_challenge_rates.csv, player_discrimination.csv,
       decision_rows.csv, threshold_chart.csv
"""
import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 60)
labeled = pd.read_csv("takes_labeled_2026.csv")
challenges = pd.read_csv("taptochallenge_challenges_2026_R.csv")

# ------------------------------------------------------------------
# Rebuild the original call and how many challenges each team had left
# ------------------------------------------------------------------
Lb = labeled.sort_values(["game_pk", "at_bat_number", "pitch_number"]).copy()
overturned = Lb["record"] == "challenged_overturned"
# Statcast stores the corrected call, so undo it to get the umpire's original call
Lb["orig_strike"] = (Lb["description"] == "called_strike") ^ overturned
Lb["chal_team"] = np.where(Lb["orig_strike"], Lb["bat_team"], Lb["fld_team"])

# a team starts with 2 and loses one for every UPHELD challenge
Lb["lost"] = (Lb["record"] == "challenged_upheld").astype(int)
grp = Lb.groupby(["game_pk", "chal_team"])["lost"]
Lb["remaining_calc"] = (2 - (grp.cumsum() - Lb["lost"])).clip(lower=0)

# extra innings: a team with 0 left gets 1
extra = Lb["inning"] >= 10
Lb["remaining"] = np.where(extra & (Lb["remaining_calc"] == 0), 1, Lb["remaining_calc"])

chk = Lb.dropna(subset=["Challenges Remaining"])
print("share matching the file's Challenges Remaining:",
      round((chk["remaining"] == chk["Challenges Remaining"]).mean(), 4))

# ------------------------------------------------------------------
# Challenge rate by how wrong the call was
# ------------------------------------------------------------------
A = Lb[Lb["dist_in"].notna() & (Lb["remaining"] > 0)].copy()
A["side"] = np.where(A["orig_strike"], "batter", "fielder")
A["margin"] = np.where(A["orig_strike"], -A["dist_in"], A["dist_in"])   # > 0 means the call was wrong
A["challenged"] = A["record"].isin(["challenged_overturned", "challenged_upheld"]).astype(int)

A["margin_bin"] = pd.cut(A["margin"], [-10, -2, -1, 0, 1, 2, 10])
tbl = A.groupby(["side", "margin_bin", "remaining"], observed=True)["challenged"].agg(["count", "mean"])
print(tbl.round(3).unstack("remaining"))

# ------------------------------------------------------------------
# Player surplus: challenges made minus challenges expected in the same situation
# ------------------------------------------------------------------
W = A[A["margin"] > 0].copy()                      # only truly wrong calls
W["value"] = W["Potential Run Expectancy delta if overturned"]
W = W.dropna(subset=["value"])
W["player"] = np.where(W["side"] == "batter", W["batter"], W["fielder_2"])
W["m_bin"] = pd.cut(W["margin"], [0, 0.5, 1, 2, 10])
W["v_bin"] = pd.qcut(W["value"], 4, labels=False, duplicates="drop")

cell = ["side", "m_bin", "v_bin", "remaining"]
W["expected"] = W.groupby(cell, observed=True)["challenged"].transform("mean")

P = (W.groupby(["side", "player"])
       .agg(chances=("challenged", "size"), made=("challenged", "sum"), expected=("expected", "sum"))
       .reset_index())
P["over"] = P["made"] - P["expected"]
k = 20                                              # shrinkage: pulls small samples toward 0
P["over_shrunk"] = P["over"] * P["chances"] / (P["chances"] + k)

nm = labeled.dropna(subset=["Player"])
bn = nm[nm["Role"] == "batter"].groupby("batter")["Player"].first()
cn = nm[nm["Role"] == "catcher"].groupby("fielder_2")["Player"].first()
P["name"] = np.where(P["side"] == "batter", P["player"].map(bn), P["player"].map(cn))
P["name"] = P["name"].fillna(P["player"].astype(str))
P.to_csv("player_challenge_rates.csv", index=False)

for side in ["batter", "fielder"]:
    S = P[(P["side"] == side) & (P["chances"] >= 30)].sort_values("over_shrunk")
    cols = ["name", "chances", "made", "expected", "over_shrunk"]
    print(side, "| players with 30+ chances:", len(S))
    print("Most below expected:"); print(S[cols].head(8).round(2).to_string(index=False))
    print("Most above expected:"); print(S[cols].tail(8).round(2).to_string(index=False))

# ------------------------------------------------------------------
# Is it a stable trait? Split games odd/even and correlate the halves
# ------------------------------------------------------------------
W["half"] = np.where(W["game_pk"] % 2 == 0, "A", "B")
H = (W.groupby(["side", "player", "half"])
       .agg(chances=("challenged", "size"), made=("challenged", "sum"), expected=("expected", "sum"))
       .reset_index())
H["surplus"] = (H["made"] - H["expected"]) / H["chances"]
Hw = H.pivot_table(index=["side", "player"], columns="half", values=["chances", "surplus"]).dropna()
for side in ["batter", "fielder"]:
    s = Hw.loc[side]
    s = s[(s[("chances", "A")] >= 15) & (s[("chances", "B")] >= 15)]
    print(side, "|", len(s), "players | split-half correlation:",
          round(s[("surplus", "A")].corr(s[("surplus", "B")]), 2))

# ------------------------------------------------------------------
# Team effect: does a player's surplus match his teammates'?
# ------------------------------------------------------------------
W["team"] = np.where(W["side"] == "batter", W["bat_team"], W["fld_team"])
T = (W.groupby(["side", "team", "player"])
       .agg(chances=("challenged", "size"), made=("challenged", "sum"), expected=("expected", "sum"))
       .reset_index())
g = T.groupby(["side", "team"])
T["tm_made"] = g["made"].transform("sum") - T["made"]
T["tm_exp"] = g["expected"].transform("sum") - T["expected"]
T["tm_ch"] = g["chances"].transform("sum") - T["chances"]
S = T[(T["chances"] >= 15) & (T["tm_ch"] >= 30)].copy()
S["surplus"] = (S["made"] - S["expected"]) / S["chances"]
S["tm_surplus"] = (S["tm_made"] - S["tm_exp"]) / S["tm_ch"]
for side in ["batter", "fielder"]:
    s = S[S["side"] == side]
    print(side, "|", len(s), "players | correlation with teammates:",
          round(s["surplus"].corr(s["tm_surplus"]), 2))

# ------------------------------------------------------------------
# Discrimination: do they challenge wrong calls more than right calls (close pitches only)?
# ------------------------------------------------------------------
R = Lb[Lb["dist_in"].notna() & (Lb["remaining"] > 0)].copy()
R["side"] = np.where(R["orig_strike"], "batter", "fielder")
R["margin"] = np.where(R["orig_strike"], -R["dist_in"], R["dist_in"])
R["challenged"] = R["record"].isin(["challenged_overturned", "challenged_upheld"]).astype(int)
R = R[R["margin"].between(-2, 2)].copy()
R["wrong"] = R["margin"] > 0
R["player"] = np.where(R["side"] == "batter", R["batter"], R["fielder_2"])
R["m_bin"] = pd.cut(R["margin"], [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
R["cnt"] = R["balls"].astype(str) + "-" + R["strikes"].astype(str)
R["expected"] = R.groupby(["side", "m_bin", "remaining", "cnt"], observed=True)["challenged"].transform("mean")


def agg(d):
    return d.groupby(["side", "player"]).agg(chances=("challenged", "size"),
                                             made=("challenged", "sum"), expected=("expected", "sum"))


def score(d, min_w, min_r):
    j = agg(d[d["wrong"]]).join(agg(d[~d["wrong"]]), lsuffix="_w", rsuffix="_r", how="inner")
    j = j[(j["chances_w"] >= min_w) & (j["chances_r"] >= min_r)].copy()
    j["wrong_surplus"] = (j["made_w"] - j["expected_w"]) / j["chances_w"]
    j["right_surplus"] = (j["made_r"] - j["expected_r"]) / j["chances_r"]
    j["discrimination"] = j["wrong_surplus"] - j["right_surplus"]
    return j


J = score(R, 15, 30).reset_index()
names = P.set_index(["side", "player"])["name"]
J["name"] = [names.get((s, p), str(p)) for s, p in zip(J["side"], J["player"])]
J.to_csv("player_discrimination.csv", index=False)

R["half"] = np.where(R["game_pk"] % 2 == 0, "A", "B")
dA = score(R[R["half"] == "A"], 8, 15)["discrimination"]
dB = score(R[R["half"] == "B"], 8, 15)["discrimination"]
both = pd.concat([dA, dB], axis=1, keys=["A", "B"]).dropna()
for side in ["batter", "fielder"]:
    s = both.loc[side]
    print(side, "discrimination split-half correlation:", round(s["A"].corr(s["B"]), 2), "|", len(s), "players")

# ------------------------------------------------------------------
# Decision rule: challenge when your confidence >= cost / (gain + cost)
# ------------------------------------------------------------------
ch = challenges.copy()
ch["gain"] = ch["Potential Run Expectancy delta if overturned"]
ch["cost"] = ch["Opp Cost RE"]
ov = ch["Result"].astype(str) == "True"
print("overturned: mean |Net RE - gain|:", round((ch.loc[ov, "Net RE"] - ch.loc[ov, "gain"]).abs().mean(), 4))
print("upheld:     mean |Net RE + cost|:", round((ch.loc[~ov, "Net RE"] + ch.loc[~ov, "cost"]).abs().mean(), 4))

D = Lb[Lb["record"].isin(["challenged_overturned", "challenged_upheld", "missed"])].copy()
D["gain"] = D["Potential Run Expectancy delta if overturned"]
D["cost"] = D["Opp Cost RE"]
D = D.dropna(subset=["gain", "cost"])
D["p_needed"] = D["cost"] / (D["gain"] + D["cost"])
D["inn_bin"] = pd.cut(D["inning"], [0, 3, 6, 8, 9, 30], labels=["1-3", "4-6", "7-8", "9", "extras"])
D["cnt"] = D["balls"].astype(str) + "-" + D["strikes"].astype(str)
D["strikes"] = D["strikes"].astype(int)
print(D.pivot_table(index="inn_bin", columns="remaining", values="p_needed",
                    aggfunc="median", observed=True).round(3))
D.to_csv("decision_rows.csv", index=False)
D.pivot_table(index=["inn_bin", "cnt"], columns="remaining", values="p_needed",
              aggfunc="median", observed=True).round(3).to_csv("threshold_chart.csv")

# ------------------------------------------------------------------
# How much run value is being captured?
# ------------------------------------------------------------------
D["side"] = np.where(D["orig_strike"], "batter", "fielder")
D["won_v"] = np.where(D["record"] == "challenged_overturned", D["gain"], 0.0)
D["lost_v"] = np.where(D["record"] == "challenged_upheld", D["cost"], 0.0)
D["left_v"] = np.where(D["record"] == "missed", D["gain"], 0.0)
tot = D[["won_v", "lost_v", "left_v"]].sum()
print("runs gained from overturns:", round(tot["won_v"], 1))
print("runs lost on upheld challenges:", round(tot["lost_v"], 1))
print("runs left on missed wrong calls:", round(tot["left_v"], 1))
print("share of available value captured:",
      round((tot["won_v"] - tot["lost_v"]) / (tot["won_v"] + tot["left_v"]), 3))

"""Step 1: pull the 2026 Statcast season and keep only the pitches that were taken.

Run:  python 01_pull_statcast.py
Takes several minutes. Writes pitches_2026.csv, takes_2026.csv, takes_2026_regular.csv
"""
import pandas as pd
from pybaseball import statcast

months = [("2026-03-25", "2026-04-30"), ("2026-05-01", "2026-05-31"),
          ("2026-06-01", "2026-06-30"), ("2026-07-01", "2026-07-31"),
          ("2026-08-01", "2026-08-31"), ("2026-09-01", "2026-09-30"),
          ("2026-10-01", "2026-10-20")]

parts = []
for start, end in months:
    print("Pulling", start, "to", end)
    parts.append(statcast(start, end))

pitches = pd.concat(parts)
pitches.to_csv("pitches_2026.csv", index=False)
print(len(pitches), "pitches")

# a "take" is a pitch the batter did not swing at
takes = pitches[pitches["description"].isin(["called_strike", "ball", "blocked_ball"])].copy()
takes.to_csv("takes_2026.csv", index=False)

# regular season only (the challenge CSVs only cover the regular season)
takes_r = takes[takes["game_type"] == "R"].copy()
takes_r.to_csv("takes_2026_regular.csv", index=False)
print(len(takes), "takes |", len(takes_r), "regular-season takes")

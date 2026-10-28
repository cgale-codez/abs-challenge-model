# ABS Challenge Model

Uses 2026 MLB Statcast data and public ABS challenge logs to study **who challenges well, which teams do, and when a challenge is worth using**, then applies the same zone math to Oregon baseball TrackMan data.

College ABS follows the MLB rules: 2 challenges per team; only the batter, pitcher or catcher can challenge. A challenge wins exactly when any part of the ball touched the zone.

## Findings (2026 MLB regular season)
- Challenge *frequency* is a stable trait (split-half r about 0.5), but batters' ability to pick the *wrong* calls is mostly noise (r = 0.09). Catchers are somewhat better (0.26).
- Teams differ little; a player's challenge rate barely matches his teammates'.
- Roughly 29% of the available run value from wrong calls was captured.
- Break-even confidence (cost / (gain + cost)) drops sharply in late innings and with 2 strikes, and is much higher with 1 challenge left than with 2.

## Setup
```
pip install -r requirements.txt
```
Python 3.10+; written and run with pandas 3.x on Windows.

## Run order (do everything inside one working folder)
| Step | Script | What it does |
|---|---|---|
| 1 | `01_pull_statcast.py` | Pulls the 2026 season with pybaseball (takes several minutes) |
| 2 | `02_label_takes.py` | Joins the challenge CSVs onto takes; computes zone distance |
| 3 | `03_player_and_team_analysis.py` | Player/team surplus, reliability, discrimination, thresholds |
| 4 | `04_simple_thresholds.py` | Cheat-sheet thresholds by inning, strikes, challenges left only |
| 5 | `05_oregon_trackman.py` | Oregon TrackMan zone scoring and film list |
| 6 | `06_mlb_comparisons.py` | MLB swing-score and wrong-call direction checks |

The three challenge CSVs (`taptochallenge_*_2026_R.csv`) come from the third-party challenge-tracking site; download them into the working folder.

## Data and privacy
CSV files and `trackman/` are git-ignored. Don't commit TrackMan data, player heights or the `oregon_*` outputs: they are private program data.

## Known limitations
- ABS zone is the bottom 27% / top 53.5% of batter height. These percentages were entered from memory and **not confirmed** against Savant's documentation.
- Run values are MLB's; college values may differ.
- Extra-innings thresholds are unreliable (small sample).
- Oregon sample is tiny (7 games, 31 hitters).

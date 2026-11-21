# makefun-gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/axitvideos/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/axitvideos/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for Grand Theft Auto VI's PlayStation 5 trophy earn rates on Exophase. `main.py` (Python 3.13, standard library only) reads one case file — the house's recorded reads of the game's trophy page up to a checkpoint — and writes five quantiles (5%, 25%, 50%, 75%, 95%) of every trophy's earn rate at the case's target, a week after launch. It treats an earn rate as the ratio it is: the tracked player pool is projected forward on its own growth curve, each trophy's earners are projected with the flow of earners gained per new tracked player measured over recent windows, and the ratio of the two is blended on the logit scale with a weighted logit-linear trend of the trophy's own rate. The spread widens with the horizon, and widens further for trophies held by only a handful of players and for histories whose pool has not moved yet.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=57bn8c8j&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>

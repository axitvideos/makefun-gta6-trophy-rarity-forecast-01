"""Forecast GTA VI trophy earn rates on Exophase, one week after launch.

Entry for make.fun's gta6-trophy-rarity-forecast-01 (program/v1, Python 3.13,
standard library only, no network). The grader runs this file once per case:
it reads the case file in the input directory (the house's recorded reads of
the game's Exophase trophy page up to the case's checkpoint) and writes one
forecast file per case to the output directory, five quantiles of every
trophy's earn rate at the case's target.

Method
------
An earn rate is a ratio: players who earned the trophy / players tracked.
The forecast builds the ratio's two sides separately, from several
candidate models, and lets the case's own history arbitrate between them.

Pool. The tracked pool N(t) grows steeply and decelerates over a launch
week. Two fits of ln N -- on ln(1 + hours since release) over the whole
history, and on plain hours over the last day -- are extrapolated to the
target and blended with weights from their fit residuals.

Candidates for a trophy's rate at the target:
- Flow. Earners gained per new tracked player (dE/dN) over recent windows,
  with window edges averaged over three reads against Exophase's step
  refreshes, shrunk toward the tier's flow when the trophy's own earner
  gains are few, and decayed as the pool grows (the marginal player earns
  less than the average one already tracked). Projected earners over the
  projected pool give the rate.
- Saturation. Earners accumulate toward a penetration ceiling as the base
  plays: rate(t) = A * (1 - exp(-t / tau)). A grid of tau values is fitted
  per trophy; this is the shape that bends a fast early climb into the
  week's plateau.
- Trend. A weighted least-squares line of the logit rate on
  ln(1 + hours since release), read at the target.
- Persistence. The last read's rate, unchanged: the honest null model.

Arbitration. Each candidate is backtested inside the history: refit on the
reads up to a cut a day (or a third of the span) before the last read, and
score its prediction of the last read on the logit scale. The median is the
candidates' logit blend weighted by prior trust times inverse backtest
error, and never sits more than 2.2 logit from the last rate.

Spread. A logit-scale sigma with a floor and a square-root-of-horizon term,
plus the blended predictor's own backtest error for that trophy, a widening
for trophies held by only a handful of players, and another for histories
whose pool never moved (the next refresh is then a jump the record has not
shown). Quantiles are the normal quantiles of that spread, rounded to 4
decimals and forced non-decreasing. Everything is a pure function of the
case file, so two runs write the same bytes.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Rates are measured from the release day's start, the same anchor the
# house's own curve baseline uses.
EPOCH = datetime(2026, 11, 19, tzinfo=timezone.utc)

NORMAL = NormalDist()

# Fallback earn rates (percent) for a trophy no read has ever listed, by tier.
TIER_PRIOR = {"platinum": 0.05, "gold": 0.4, "silver": 1.2, "bronze": 2.5}

# Saturation time constants (hours) tried per trophy.
TAU_GRID = (16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0)

# How fast the flow of earners per new player decays as the pool grows:
# m(N) = m_now * (N / N_now) ** (-FLOW_DECAY). Tuned on synthetic launch
# histories (rising, saturating and dilution-falling worlds): 0.28 split
# the difference between the rising worlds' preference for a slower decay
# and the falling world's for a faster one.
FLOW_DECAY = 0.28


def parse_time(text):
    """ISO 8601 UTC text to an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_epoch(moment):
    return (moment - EPOCH).total_seconds() / 3600.0


def clamp_rate(pct):
    """The scorer clamps rates to 0.01..99.99 before the logit; so do we."""
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def fit_line(xs, ys, ws):
    """Weighted least squares y = a + b*x.

    Returns (a, b, weighted residual variance) or None.
    """
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = my - b * mx
    var = sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / sw
    return (a, b, var)


def project_players(reads, h_target):
    """Project the tracked pool at the target hour.

    reads: list of (hour, players) with players > 0, oldest first.
    Returns (n_target, growth) with growth = n_target / n_now >= 1.
    """
    n_now = float(reads[-1][1])
    views = []
    fit = fit_line([math.log1p(h) for h, _ in reads],
                   [math.log(n) for _, n in reads], [1.0] * len(reads))
    if fit is not None:
        a, b, var = fit
        views.append((math.exp(a + b * math.log1p(h_target)), var))
    recent = [(h, n) for h, n in reads if h >= reads[-1][0] - 24.0]
    if len(recent) >= 2 and recent[-1][0] > recent[0][0]:
        fit2 = fit_line([h for h, _ in recent],
                        [math.log(n) for _, n in recent], [1.0] * len(recent))
        if fit2 is not None:
            a2, b2, var2 = fit2
            views.append((math.exp(a2 + b2 * h_target), var2))
    if not views:
        return n_now, 1.0
    # The view that tracked the recorded pool better earns the larger say.
    weights = [1.0 / (var + 0.02) for _, var in views]
    log_n = sum(w * math.log(max(n, 1.0)) for (n, _), w in zip(views, weights))
    n_target = math.exp(log_n / sum(weights))
    n_target = min(max(n_target, n_now), 120.0 * n_now)
    return n_target, n_target / n_now


def edge_state(chunk, idx):
    """Mean pool and mean earners of trophy idx over up to three reads.

    Exophase refreshes in steps, so a window edge taken from a single read
    can sit just before a jump; averaging three reads softens that.
    Returns (players, earners) or None when no read in the chunk lists
    the trophy.
    """
    ns = []
    es = []
    for _, n, rates in chunk:
        if rates[idx] is None:
            continue
        ns.append(float(n))
        es.append(rates[idx] * n / 100.0)
    if not ns:
        return None
    return (sum(ns) / len(ns), sum(es) / len(es))


def flow_estimate(reads, idx, hours_span):
    """Earners gained per new tracked player for trophy idx, in percent.

    reads: list of (hour, players, rates). Returns (flow, earners_gained)
    or None when the pool did not move inside the window.
    """
    if len(reads) < 2:
        return None
    h_last = reads[-1][0]
    window = [r for r in reads if r[0] >= h_last - hours_span]
    if len(window) < 2:
        window = reads[-2:]
    first = edge_state(window[:3], idx)
    last = edge_state(window[-3:], idx)
    if first is None or last is None:
        return None
    dn = last[0] - first[0]
    if dn <= 0:
        return None
    de = last[1] - first[1]
    return (min(100.0, max(0.0, 100.0 * de / dn)), de)


def flow_candidate(reads, idx, n_target, tier_flow):
    """Rate at the target from the decayed-flow projection, or None."""
    if len(reads) < 2:
        return None
    n_now = float(reads[-1][1])
    r_now = None
    for _, _, rates in reversed(reads):
        if rates[idx] is not None:
            r_now = rates[idx]
            break
    if r_now is None:
        return None
    parts = []
    for span, weight in ((18.0, 0.65), (48.0, 0.35)):
        est = flow_estimate(reads, idx, span)
        if est is not None:
            parts.append((est[0], est[1], weight))
    if not parts:
        return None
    flow = 0.0
    wsum = 0.0
    for m, gained, weight in parts:
        if tier_flow is not None:
            # Shrink toward the tier's flow when this trophy's own earner
            # gains are few enough to be mostly luck.
            trust = gained / (gained + 20.0) if gained > 0 else 0.0
            m = trust * m + (1.0 - trust) * tier_flow
        flow += m * weight
        wsum += weight
    flow /= wsum
    growth = n_target / n_now
    earners_now = r_now * n_now / 100.0
    if growth <= 1.0:
        earners_target = earners_now
    elif FLOW_DECAY >= 1.0:
        earners_target = earners_now + (flow / 100.0) * n_now * math.log(growth)
    else:
        earners_target = earners_now + (flow / 100.0) * n_now * (
            growth ** (1.0 - FLOW_DECAY) - 1.0) / (1.0 - FLOW_DECAY)
    return min(99.99, max(0.01, 100.0 * earners_target / n_target))


def saturation_candidate(reads, idx, h_target):
    """Rate at the target from a penetration-ceiling fit, or None."""
    pts = [(h, n, rates[idx]) for h, n, rates in reads if rates[idx] is not None]
    if len(pts) < 4:
        return None
    h_last = pts[-1][0]
    recent = [p for p in pts if p[0] >= h_last - 48.0] or pts[-4:]
    best = None
    for tau in TAU_GRID:
        num = 0.0
        den = 0.0
        for h, n, r in recent:
            f = 1.0 - math.exp(-max(h, 0.05) / tau)
            if f < 0.02:
                continue
            earners = r * n / 100.0
            num += earners / f
            den += n
        if den <= 0:
            continue
        ceiling = num / den  # penetration ceiling as a fraction
        if ceiling <= 0 or ceiling > 0.995:
            continue
        err = 0.0
        for h, n, r in recent:
            pred = 100.0 * ceiling * (1.0 - math.exp(-max(h, 0.05) / tau))
            err += (logit(pred) - logit(r)) ** 2
        err /= len(recent)
        if best is None or err < best[0]:
            best = (err, ceiling, tau)
    if best is None:
        return None
    _, ceiling, tau = best
    pred = 100.0 * ceiling * (1.0 - math.exp(-max(h_target, 0.05) / tau))
    return min(99.99, max(0.01, pred))


def trend_candidate(series, h_target):
    """Logit-linear trend of one trophy's rate, read at the target hour."""
    if len(series) < 3:
        return None
    xs = [math.log1p(h) for h, _ in series]
    ys = [logit(r) for _, r in series]
    h_last = series[-1][0]
    ws = [1.0 + 2.0 * (h / h_last if h_last > 0 else 0.0) for h, _ in series]
    fit = fit_line(xs, ys, ws)
    if fit is None:
        return None
    a, b, _ = fit
    b = min(3.0, max(-3.0, b))
    return inv_logit(a + b * math.log1p(h_target))


def candidates(reads, idx, trophy, n_target, h_target, tier_flow):
    """All candidate rates (percent) for one trophy, keyed by name."""
    out = {}
    r_last = None
    for _, _, rates in reversed(reads):
        if rates[idx] is not None:
            r_last = rates[idx]
            break
    if r_last is not None:
        out["persistence"] = clamp_rate(r_last)
    flow = flow_candidate(reads, idx, n_target, tier_flow)
    if flow is not None:
        out["flow"] = flow
    sat = saturation_candidate(reads, idx, h_target)
    if sat is not None:
        out["saturation"] = sat
    series = [(h, rates[idx]) for h, _, rates in reads if rates[idx] is not None]
    trend = trend_candidate(series, h_target)
    if trend is not None:
        out["trend"] = trend
    return out


# Prior trust per candidate, before the backtest speaks.
PRIOR = {"flow": 1.0, "saturation": 1.0, "trend": 0.6, "persistence": 0.8}


def backtest_weights(reads, idx, trophy, tier_flow):
    """Weight per candidate from an in-history backtest.

    Refit every candidate on the reads up to a cut before the last read
    and score its prediction of the last read's rate on the logit scale.
    Returns (weights, blended backtest error) or (None, None) when the
    history is too short or the pool never moved before the cut.
    """
    if len(reads) < 6:
        return None, None
    h_first, h_last = reads[0][0], reads[-1][0]
    span = h_last - h_first
    if span <= 0:
        return None, None
    cut = h_last - min(24.0, span / 3.0)
    head = [r for r in reads if r[0] <= cut]
    if len(head) < 4:
        return None, None
    actual = None
    for _, _, rates in reversed(reads):
        if rates[idx] is not None:
            actual = rates[idx]
            break
    if actual is None:
        return None, None
    n_at_last, _ = project_players([(h, n) for h, n, _ in head if n > 0], h_last)
    preds = candidates(head, idx, trophy, n_at_last, h_last, tier_flow)
    if not preds:
        return None, None
    weights = {}
    errs = {}
    for name, pred in preds.items():
        err = abs(logit(pred) - logit(actual))
        errs[name] = err
        weights[name] = PRIOR.get(name, 0.7) / (err * err + 0.09)
    wsum = sum(weights.values())
    blend_err = sum(weights[k] * errs[k] for k in weights) / wsum
    return weights, blend_err


def tier_flows(reads, trophies):
    """Mean flow (percent) per tier over the 48-hour window."""
    sums = {}
    for idx, trophy in enumerate(trophies):
        est = flow_estimate(reads, idx, 48.0)
        if est is None:
            continue
        tier = trophy.get("tier")
        total, count = sums.get(tier, (0.0, 0))
        sums[tier] = (total + est[0], count + 1)
    return {tier: total / count for tier, (total, count) in sums.items() if count}


def forecast_trophy(case, reads, idx, trophy, n_target, growth, h_target,
                    tier_flow):
    """Five quantiles (percent) for one trophy."""
    horizon_days = case["horizon_minutes"] / 1440.0
    series = [(h, rates[idx]) for h, _, rates in reads if rates[idx] is not None]
    if not series:
        return quantiles(TIER_PRIOR.get(trophy.get("tier"), 1.0), 1.6)
    _, r_now = series[-1]
    n_now = float(reads[-1][1])
    earners_now = r_now * n_now / 100.0

    preds = candidates(reads, idx, trophy, n_target, h_target, tier_flow)
    weights, blend_err = backtest_weights(reads, idx, trophy, tier_flow)
    if weights is None:
        weights = {name: PRIOR.get(name, 0.7) for name in preds}
    else:
        weights = {name: weights.get(name, 0.25 * PRIOR.get(name, 0.7))
                   for name in preds}
    wsum = sum(weights.values())
    med_logit = sum(weights[name] * logit(pred) for name, pred in preds.items())
    med_logit /= wsum
    # Never let the blend run further than 2.2 logit from the last rate:
    # a week is long, but it is not a different game.
    med_logit = min(logit(r_now) + 2.2, max(logit(r_now) - 2.2, med_logit))
    median = inv_logit(med_logit)

    sigma = spread_sigma(horizon_days, earners_now, "flow" in preds, growth,
                         blend_err)
    return quantiles(median, sigma)


def spread_sigma(horizon_days, earners_now, has_flow, growth, blend_err):
    """Logit-scale spread of a trophy's forecast."""
    var = 0.30 ** 2 + (0.30 * math.sqrt(horizon_days)) ** 2
    if blend_err is not None:
        var += (0.8 * blend_err) ** 2
    if earners_now < 40:
        # A trophy held by a handful of players moves in jumps.
        var += (0.30 * math.exp(-earners_now / 40.0)) ** 2
    if not has_flow:
        # No measured flow: the next refresh is an unseen jump.
        var += 0.35 ** 2
    if growth < 1.2:
        # A pool that has stopped growing leaves progression as the only
        # mover, and it is the part the record shows least directly.
        var += 0.15 ** 2
    return min(2.2, max(0.35, math.sqrt(var)))


def quantiles(median, sigma):
    med = logit(median)
    out = []
    for tau in (0.05, 0.25, 0.5, 0.75, 0.95):
        q = inv_logit(med + NORMAL.inv_cdf(tau) * sigma)
        out.append(round(min(100.0, max(0.0, q)), 4))
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def forecast_case(case):
    reads = []
    for read in case["history"]:
        moment = parse_time(read["at"])
        reads.append((hours_since_epoch(moment), read["players_tracked"],
                      read["rates"]))
    h_target = hours_since_epoch(parse_time(case["target"]))
    n_target, growth = project_players(
        [(h, n) for h, n, _ in reads if n and n > 0], h_target)
    flows = tier_flows(reads, case["trophies"])
    rows = []
    for idx, trophy in enumerate(case["trophies"]):
        q = forecast_trophy(case, reads, idx, trophy, n_target, growth,
                            h_target, flows.get(trophy.get("tier")))
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

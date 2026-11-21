# city.py - the worlds (Miami from my fieldwork; Philadelphia and New Orleans from Yelp), the night clocks,
# and how a group decides. simulate.py and irl.py both import this, so they agree on how a night works.
import os
import numpy as np
import pandas as pd

# ---------------- settings you can turn ----------------
TEMPERATURE = 1.5   # rationality: 1 = careful choosers, higher = noisier picks. 1.5 = medium-low
HORIZON = 2         # moves a group looks ahead: 1 = pure reaction, 2 = loose plan (fieldwork: 1-2 on average)

FEATURES = ["cover", "guest list", "table", "DJ / vibe", "vibe late (escalation)", "door",
            "moving", "hotspot (tourists)", "hotspot (locals)", "home when tired"]
K = len(FEATURES)
HOURS = ["10pm", "11pm", "12am", "1am", "2am", "3am", "4am", "5am", "6am", "7am"]
H = len(HOURS)
E_MAX = 6                                   # group energy: 6 = fresh, 0 = crashed
PATHS = ["early ticket", "late ticket", "guest list", "table"]   # set by the group's anchor
BUDGET = [80, 600, 200, 3000]               # $ per person each path is comfortable spending
TYPES = ["local", "tourist"]
S = len(PATHS) * len(TYPES)                 # stratum s = path * 2 + type (fixed for the night)

# Ground truth (hidden from irl.py). Order follows my fieldwork ranking: cover > guest list > table ~ DJ > door
TRUE_WEIGHTS = {  # cover  list  table  DJ  late  door  move  hot(T) hot(L) home
    "Miami weekend":   [-2.0, 1.6, 1.1, 1.1, 1.2, -0.5, -0.8, 1.5, 0.3, 1.0],
    "Miami weeknight": [-2.4, 1.9, 1.0, 1.0, 0.5, -0.5, -1.0, 1.2, 0.3, 1.6],
    "Philadelphia":    [-2.0, 1.4, 1.0, 1.0, 0.2, -0.5, -1.0, 1.2, 0.4, 2.2],
    "New Orleans":     [-1.8, 1.4, 1.0, 1.1, 1.4, -0.5, -0.8, 1.4, 0.4, 0.9],
    "Late-start city": [-1.8, 1.5, 1.1, 1.0, 1.6, -0.5, -0.7, 1.3, 0.4, 0.8],
}
CONTEXTS = list(TRUE_WEIGHTS)

# Night clocks: start (hour index), price scale, whether hype drains extra energy, how late the vibe peaks,
# tourist share, and how often members hold each access path (early ticket, late ticket, guest list, table).
# In Philadelphia and New Orleans, venues and closing times come from Yelp, and most people just walk into bars.
MIAMI_PATHS, BAR_PATHS = [0.45, 0.30, 0.20, 0.05], [0.85, 0.07, 0.06, 0.02]
CLOCKS = {
    "Miami weekend":   dict(world="Miami", start=(0, 2), price=1.0, drain_hi=0, peak_shift=0, tourists=0.5, paths=MIAMI_PATHS),
    "Miami weeknight": dict(world="Miami", start=(0, 2), price=0.6, drain_hi=1, peak_shift=-1, tourists=0.4, paths=MIAMI_PATHS),
    "Philadelphia":    dict(world="Philadelphia", start=(0, 1), price=1.0, drain_hi=1, peak_shift=0, tourists=0.2, paths=BAR_PATHS),
    "New Orleans":     dict(world="New Orleans", start=(0, 2), price=1.0, drain_hi=0, peak_shift=0, tourists=0.5, paths=BAR_PATHS),
    "Late-start city": dict(world="Miami", start=(1, 3), price=0.8, drain_hi=0, peak_shift=2, tourists=0.3, paths=MIAMI_PATHS),
}

# ---------------- Miami: 18 venues in 4 neighborhoods (archetypes from my fieldwork) ----------------
# name, hood, early $, late $, table $, promoter?, door 0-1, popularity 0-1, vibe 0-1, peak hour, closes (hour idx), open weeknights?
# closes: 4 = 2am, 5 = 3am, 7 = 5am, 10 = never (runs into the morning / 24h)
V = [("WY courtyard bar", "Wynwood", 20, 150, 0, 1, .1, .5, .6, 2, 4, 1),
     ("WY brewery patio", "Wynwood", 20, 150, 0, 0, .1, .3, .5, 1, 4, 1),
     ("WY rooftop", "Wynwood", 30, 200, 4000, 1, .3, .6, .7, 3, 5, 1),
     ("WY dive club", "Wynwood", 20, 150, 0, 1, .1, .4, .8, 3, 5, 1),
     ("WY gallery party", "Wynwood", 25, 180, 0, 0, .2, .4, .7, 2, 4, 0),
     ("BK lounge", "Brickell", 50, 400, 8000, 1, .7, .6, .7, 3, 7, 1),
     ("BK supper club", "Brickell", 60, 600, 15000, 0, .8, .5, .6, 2, 7, 1),
     ("BK sky club", "Brickell", 60, 1000, 30000, 1, .9, .7, .9, 4, 7, 0),
     ("BK speakeasy", "Brickell", 50, 400, 0, 0, .6, .4, .5, 2, 7, 1),
     ("SB megaclub", "South Beach", 50, 300, 20000, 1, .7, 1.0, 1.0, 4, 7, 0),
     ("SB hotel pool party", "South Beach", 40, 200, 6000, 1, .5, .8, .8, 2, 5, 1),
     ("SB Lincoln Rd club", "South Beach", 30, 150, 4000, 1, .5, .7, .8, 3, 7, 1),
     ("SB beach bar", "South Beach", 30, 150, 0, 0, .2, .6, .6, 2, 7, 1),
     ("SB lounge", "South Beach", 40, 200, 6000, 0, .6, .5, .7, 3, 7, 0),
     ("DT 24h club", "Downtown", 60, 300, 10000, 1, .6, .9, .9, 6, 10, 1),
     ("DT warehouse (till morning)", "Downtown", 40, 200, 0, 0, .4, .8, 1.0, 7, 10, 0),
     ("DT bar row", "Downtown", 30, 150, 0, 1, .2, .4, .6, 2, 5, 1),
     ("DT late club", "Downtown", 40, 250, 5000, 1, .5, .6, .8, 5, 7, 1)]
N = len(V)                   # every city gets 18 venues, so the tables have the same shape everywhere
HOME = N                     # location N = pregame / start; action N = go home
L, A = N + 1, N + 1
CENTER = {"Wynwood": (0, 2.0), "Downtown": (0, 0.6), "Brickell": (0, -0.6), "South Beach": (4, 0.8)}
rng = np.random.default_rng(7)
COLS = ["early", "late", "table", "promo", "door", "pop", "vibe", "peak", "close", "weeknight"]
MIAMI = dict(names=[v[0] for v in V], hood=[v[1] for v in V],
             **{k: np.array([v[i + 2] for v in V], float) for i, k in enumerate(COLS)},
             xy=np.array([CENTER[v[1]] for v in V]) + rng.normal(0, 0.2, (N, 2)),
             pregame={0: np.array([0.3, -0.4]), 1: np.array([4.2, 0.6])})   # locals mainland, tourists in Beach hotels


def yelp_world(name):
    """Philadelphia / New Orleans from data/yelp_*_venues.csv (made by yelp.py). Yelp gives the venues, location,
    price tier, dancing / music / ambience tags, Saturday closing time and check-ins by hour. Cover, tables and
    promoters aren't in Yelp, so they follow simple rules from price tier and dancing (bars rarely charge cover)."""
    d = pd.read_csv(f"data/yelp_{name.lower().replace(' ', '_')}_venues.csv")
    late_rank = d.late_share.rank(pct=True).values           # which venues stay busy late, relative to the city
    km = np.c_[d.lat * 111, d.lon * 111 * np.cos(np.radians(d.lat.mean()))]
    xy = (km - km.mean(0)) / 3                               # 3 km ~ one Miami unit of travel hassle
    close = np.clip(d.close_h.values - 22, 0, 10)            # hour index: 2am -> 4, 5am -> 7, 24h -> 10
    early = 5.0 * d.dance.values + 5 * (d.price.values - 1)
    busiest = d.hood[d.reviews.idxmax()]                     # tourists start where the most-reviewed spot is
    return dict(names=list(d.name), hood=list(d.hood), early=early, late=early + 10 * d.dance.values,
                table=np.where((d.dance == 1) & (d.price >= 3), 500.0 * d.price, 0.0), promo=d.dance.values * 1.0,
                door=np.clip(0.1 + 0.15 * (d.price.values - 1) + 0.2 * d.dance.values, 0, 1),
                pop=0.2 + 0.8 * d.reviews.rank(pct=True).values,
                vibe=np.clip(0.3 + 0.2 * d.dance + 0.15 * d.dj_or_live + 0.1 * d.trendy + 0.25 * late_rank, 0, 1).values,
                peak=np.minimum(1 + np.round(3 * late_rank), close - 1), close=close, weeknight=np.ones(N),
                xy=xy, pregame={0: np.zeros(2), 1: xy[(d.hood == busiest).values].mean(0)},
                checkins=d[[f"c{h:02d}" for h in [22, 23, 0, 1, 2, 3, 4, 5, 6, 7]]].values)


WORLDS = {"Miami": MIAMI}
for _name in ["Philadelphia", "New Orleans"]:
    if os.path.exists(f"data/yelp_{_name.lower().replace(' ', '_')}_venues.csv"):
        WORLDS[_name] = yelp_world(_name)
    else:
        CONTEXTS.remove(_name)                                # run python yelp.py once to add the real cities


def world(context):
    return WORLDS[CLOCKS[context]["world"]]


def closing_times(context):
    W = world(context)
    close = W["close"].copy()
    if context == "Miami weeknight":
        close = np.where(W["weeknight"] == 1, close, 0)       # some big clubs only run weekends
    if context == "Late-start city":
        close = np.minimum(close + 2, 10)                     # Madrid-style: it all runs later
    return close


def build_tables(context):
    """Everything a group could do, everywhere, every hour.
    Phi[s,h,l,e,a,k] = features of action a; ok[s,h,l,e,a] = allowed?; ne[h,e,a] = energy after."""
    c, W = CLOCKS[context], world(context)
    close = closing_times(context)
    EARLY, LATE, TABLE, PROMO, DOOR, POP, VIBE, PEAK, XY = (W[k] for k in
                                                            ["early", "late", "table", "promo", "door", "pop", "vibe", "peak", "xy"])
    hrs = np.arange(H)
    vibe = VIBE[:, None] * np.exp(-0.5 * ((hrs[None, :] - (PEAK[:, None] + c["peak_shift"])) / 2.0) ** 2)   # [venue, hour]
    drain = 1 + c["drain_hi"] * (vibe > 0.6)                         # hype burns extra energy in some cities
    ne = np.zeros((H, E_MAX + 1, A), int)
    for e in range(E_MAX + 1):
        ne[:, e, :N] = np.maximum(e - drain.T, 0)
    Phi = np.zeros((S, H, L, E_MAX + 1, A, K), np.float32)
    ok = np.zeros((S, H, L, E_MAX + 1, A), bool)
    for s in range(S):
        p, g = s // 2, s % 2
        for h in range(H):
            for l in range(L):
                here = W["pregame"][g] if l == HOME else XY[l]
                for a in range(N):
                    if h >= close[a] or h == H - 1:
                        continue                                    # closed, or the night is over
                    enter = a != l
                    early, late, table = EARLY[a] * c["price"], LATE[a] * c["price"], TABLE[a] * c["price"]
                    cost = [early if h <= 1 else late, late, 10 if PROMO[a] and h <= 2 else late,
                            table / 5 if TABLE[a] > 0 else late][p]
                    f = [min(cost / BUDGET[p], 2.0) if enter else 0,
                         float(p == 2 and PROMO[a] == 1), float(p == 3 and TABLE[a] > 0),
                         vibe[a, h], vibe[a, h] * h / (H - 1),
                         DOOR[a] * (0.3 if p >= 2 else 1.0) if enter else 0,
                         1 + np.linalg.norm(here - XY[a]) if enter else 0,   # leaving costs something, plus the trip
                         POP[a] * (g == 1), POP[a] * (g == 0), 0]
                    Phi[s, h, l, 1:, a] = f                         # energy 0 = crashed, can't go anywhere
                    ok[s, h, l, 1:, a] = True
    Phi[:, :, :, :, HOME, 9] = (E_MAX - np.arange(E_MAX + 1)) / E_MAX   # going home feels better the more tired
    ok[..., HOME] = True
    return Phi, ok, ne


def q_values(w, tables, horizon=HORIZON, temp=TEMPERATURE):
    """Soft planning with a limited lookahead. Returns Q[t] (value of each action at hour t)
    and dQ[t] (how Q changes with each weight - this is what makes the fitting exact)."""
    Phi, ok, ne = tables
    R = np.where(ok, Phi @ np.asarray(w, np.float32), -np.inf)
    aa = np.arange(N)[None, :]

    def backup(h, V, F):                  # value of acting at hour h, then continuing with (V, F)
        Q, dQ = R[:, h].copy(), Phi[:, h].copy()
        if V is not None:
            ee = ne[h][:, :N]
            Q[..., :N] += V[:, aa, ee][:, None]
            dQ[..., :N, :] += F[:, aa, ee][:, None]
        return np.where(ok[:, h], Q, -np.inf), dQ

    def soft(Q, dQ):                      # soft max over actions: value + expected features
        V = temp * np.logaddexp.reduce(Q / temp, axis=-1)
        pi = np.exp((Q - V[..., None]) / temp)
        return V, (pi[..., None] * dQ).sum(-2)

    Qs, dQs = [None] * H, [None] * H
    if horizon >= H:                      # full planner: one backward sweep
        V = F = None
        for t in reversed(range(H)):
            Qs[t], dQs[t] = backup(t, V, F)
            V, F = soft(Qs[t], dQs[t])
        return Qs, dQs
    for t in range(H):
        V = F = None
        end = t + horizon
        if horizon > 0:
            if end >= H - 1:
                V, F = soft(*backup(H - 1, None, None))
                end = H - 1
            else:                         # past the horizon, assume you'd head home
                V, F = R[:, end, :, :, HOME], Phi[:, end, :, :, HOME]
            for j in range(end - 1, t, -1):
                V, F = soft(*backup(j, V, F))
        Qs[t], dQs[t] = backup(t, V, F)
    return Qs, dQs

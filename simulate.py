# simulate.py - generate nights out from groups with KNOWN preferences, under five night clocks
# (Miami weekend / weeknight, Philadelphia and New Orleans from Yelp, a Madrid-style late-start city).
# Output: data/nights.csv (one row per decision) and data/venues.csv
import os
import numpy as np
import pandas as pd
import city

rng = np.random.default_rng(42)
NIGHTS_PER_CONTEXT = 1000
RUNNER_RATE = 0.4            # fieldwork: about 2 in 5 nights, someone leaves early on purpose
os.makedirs("data", exist_ok=True)

rows, night = [], 0
for context in city.CONTEXTS:
    tables = city.build_tables(context)
    Qs, _ = city.q_values(city.TRUE_WEIGHTS[context], tables)
    clock, W = city.CLOCKS[context], city.world(context)
    for _ in range(NIGHTS_PER_CONTEXT):
        size = int(rng.integers(3, 7))
        paths = rng.choice(4, size=size, p=clock["paths"])   # what each member can get in on
        p = int(paths.max())                      # the anchor: whoever has the table, then the promoter, then the budget
        g = int(rng.random() < clock["tourists"])
        s = p * 2 + g
        lo, hi = clock["start"]
        h = lo if p == 0 else int(rng.integers(min(lo + 1, hi), hi + 1))   # early tickets arrive first
        energy = int(round(rng.integers(city.E_MAX - 2, city.E_MAX + 1, size).mean()))   # members differ
        runner_at = int(rng.integers(h + 1, h + 4)) if rng.random() < RUNNER_RATE else -1
        loc = city.HOME
        while h < city.H:
            q = Qs[h][s, loc, energy]
            pi = np.exp((q - q.max()) / city.TEMPERATURE)
            action = int(rng.choice(city.A, p=pi / pi.sum()))
            rows.append([night, context, city.TYPES[g], city.PATHS[p], s, size, h, loc, energy, action,
                         "go home" if action == city.HOME else W["names"][action], int(h == runner_at)])
            if action == city.HOME:
                break
            loc, energy, h = action, tables[2][h, energy, action], h + 1
        night += 1

cols = ["night", "context", "type", "path", "stratum", "group_size", "hour", "location", "energy",
        "action", "action_name", "runner_left"]
df = pd.DataFrame(rows, columns=cols)
df.to_csv("data/nights.csv", index=False)
venue_rows = []
for name, W in city.WORLDS.items():
    for i in range(city.N):
        venue_rows.append([name, W["names"][i], W["hood"][i], W["early"][i], W["late"][i], W["table"][i], W["promo"][i],
                           W["door"][i], round(W["pop"][i], 2), round(W["vibe"][i], 2),
                           city.HOURS[int(W["close"][i])] if W["close"][i] < 10 else "morning"])
pd.DataFrame(venue_rows, columns=["world", "venue", "neighborhood", "early_$", "late_$", "table_$", "promoter", "door",
                                  "popularity", "vibe", "closes"]).to_csv("data/venues.csv", index=False)

# quick sanity checks on the behaviour from fieldwork
v = df[df.action != city.HOME].copy()
print(f"{df.night.nunique()} nights, {len(df)} decisions")
print("hours out per night:", v.groupby("context").size().div(df.groupby("context").night.nunique()).round(2).to_dict())
stay = v[v.location != city.HOME].copy()   # first entry of the night is not a hop
stay["hood"] = [city.world(c)["hood"][l] for c, l in zip(stay.context, stay.location)]
print("after arriving, chance the next hour is spent somewhere new (hop vs settle):")
print((stay.action != stay.location).groupby([stay.context, stay.hood]).mean().round(2).to_string())
mw = v[v.context == "Miami weekend"]
top = mw.groupby("action").size().nlargest(3).index
print("Miami weekend share of time at the top-3 hotspots:", mw.assign(hot=mw.action.isin(top)).groupby("type").hot.mean().round(2).to_dict())
print("nights with a runner:", round(df.groupby("night").runner_left.max().mean(), 2))

# calibration check: do simulated ARRIVALS at each real venue line up with its Yelp check-ins (10pm-7am)?
# (a check-in happens when you walk in, so compare walking in, not hours spent inside)
for c in ["Philadelphia", "New Orleans"]:
    if c not in city.CONTEXTS:
        continue
    d = v[(v.context == c) & (v.action != v.location)]
    sim = np.zeros((city.N, city.H))
    np.add.at(sim, (d.action.values, d.hour.values), 1)
    real = city.world(c)["checkins"].astype(float)
    r_venue = np.corrcoef(sim.sum(1), real.sum(1))[0, 1]                 # which venues draw the crowd
    r_hour = np.corrcoef(sim.sum(0), real.sum(0))[0, 1]                  # when the city is busy
    r_all = np.corrcoef(sim.ravel(), real.ravel())[0, 1]                 # venue x hour together
    print(f"calibration {c}: venue r = {r_venue:.2f}, hour-of-night r = {r_hour:.2f}, venue x hour r = {r_all:.2f}")

# yelp.py - turn the Yelp Open Dataset into two real cities for the model: Philadelphia (2am last call)
# and New Orleans (the late city). Reads the big raw files ONCE and writes small CSVs that city.py uses:
#   data/yelp_philadelphia_venues.csv, data/yelp_new_orleans_venues.csv  (18 venues each, 4 neighborhoods)
#   data/yelp_open_by_hour.csv  (share of ALL the city's bars open at each hour of a Saturday night)
# Usage: put yelp_academic_dataset_business.json and yelp_academic_dataset_checkin.json in data/yelp/
#        python yelp.py
# Data: Yelp Open Dataset (yelp.com/dataset), used under its terms for educational / non-commercial use.
import ast
import json
import os
import numpy as np
import pandas as pd

YELP_DIR = "data/yelp"
BUSINESS = os.path.join(YELP_DIR, "yelp_academic_dataset_business.json")
CHECKIN = os.path.join(YELP_DIR, "yelp_academic_dataset_checkin.json")
CITIES = {"Philadelphia": ("PA", -5), "New Orleans": ("LA", -6)}   # state, UTC offset (standard time; DST ignored)
VENUES_PER_CITY = 18
ZIP_NAMES = {"19106": "Old City", "19107": "Center City East", "19103": "Center City / Rittenhouse", "19102": "Center City",
             "19123": "Northern Liberties", "19125": "Fishtown", "19122": "Fishtown / Kensington",
             "19147": "Queen Village / South St", "19146": "Graduate Hospital", "19104": "University City",
             "19130": "Fairmount", "19148": "South Philly", "19127": "Manayunk", "19140": "North Philly",
             "19114": "Northeast Philly", "19135": "Northeast Philly",
             "70116": "French Quarter / Marigny", "70112": "CBD", "70130": "French Quarter / CBD",
             "70117": "Bywater", "70115": "Uptown", "70118": "Uptown", "70119": "Mid-City", "70113": "Central City"}


def hours_on(text):
    """'16:0-2:0' -> (16, 26): open hour and close hour on one clock (past midnight = 24+).
    '0:0-0:0' -> (0, 48): Yelp's way of saying open 24 hours."""
    o, c = [int(t.split(":")[0]) + int(t.split(":")[1]) / 60 for t in text.split("-")]
    if o == 0 and c == 0:
        return 0.0, 48.0
    return o, c + 24 if c <= o else c


def parse(value):
    """Yelp stores attributes like Music as strings of Python dicts: "{'dj': True, ...}"."""
    try:
        return ast.literal_eval(value) if isinstance(value, str) else value
    except (ValueError, SyntaxError):
        return None


def build():
    if not (os.path.exists(BUSINESS) and os.path.exists(CHECKIN)):
        print(f"Yelp files not found in {YELP_DIR}/ - download the Yelp Open Dataset first (see README)")
        return
    # 1) every open bar / club in the two cities with Saturday hours
    rows = []
    with open(BUSINESS) as f:
        for line in f:
            b = json.loads(line)
            cats = b.get("categories") or ""
            if b["city"] not in CITIES or b["state"] != CITIES[b["city"]][0] or b.get("is_open") != 1:
                continue
            if not any(k in cats for k in ["Bars", "Nightlife", "Dance Clubs"]) or "Hotels" in cats or "Taxi" in cats:
                continue
            sat = (b.get("hours") or {}).get("Saturday")
            if not sat:
                continue
            attr = b.get("attributes") or {}
            music, amb = parse(attr.get("Music")) or {}, parse(attr.get("Ambience")) or {}
            o, c = hours_on(sat)
            price = parse(attr.get("RestaurantsPriceRange2"))
            rows.append({"city": b["city"], "business_id": b["business_id"], "name": b["name"], "zip": b.get("postal_code", ""),
                         "lat": b["latitude"], "lon": b["longitude"], "reviews": b["review_count"],
                         "price": int(price) if price else 2, "dance": int("Dance Clubs" in cats or parse(attr.get("GoodForDancing")) is True),
                         "dj_or_live": int(bool(music.get("dj") or music.get("live"))),
                         "trendy": int(bool(amb.get("trendy") or amb.get("hipster"))), "open_h": o, "close_h": c,
                         "restaurant": int("Restaurants" in cats)})
    bars = pd.DataFrame(rows)
    # 2) check-ins by LOCAL hour (Yelp timestamps are UTC)
    keep = dict(zip(bars.business_id, bars.city))
    busy = {}
    with open(CHECKIN) as f:
        for line in f:
            c = json.loads(line)
            if c["business_id"] in keep:
                counts = np.zeros(24, int)
                for stamp in c["date"].split(", "):
                    counts[(int(stamp[11:13]) + CITIES[keep[c["business_id"]]][1]) % 24] += 1
                busy[c["business_id"]] = counts
    hourly = np.array([busy.get(i, np.zeros(24, int)) for i in bars.business_id])
    for h in range(24):
        bars[f"c{h:02d}"] = hourly[:, h]
    late = hourly[:, [22, 23, 0, 1, 2, 3]].sum(1)
    bars["late_share"] = np.where(hourly.sum(1) > 0, late / np.maximum(hourly.sum(1), 1), 0)
    # 3) share of all bars open at each hour of Saturday night (8pm to 7am) - the real night clock
    clock = [[city, t] + [((g.open_h <= t) & (g.close_h > t)).mean()] for city, g in bars.groupby("city") for t in range(20, 32)]
    pd.DataFrame(clock, columns=["city", "hour", "share_open"]).to_csv("data/yelp_open_by_hour.csv", index=False)
    # 4) pick 18 venues per city: 4 neighborhoods by k-means on location, most-reviewed venues in each
    rng = np.random.default_rng(3)
    for city, g in bars.groupby("city"):
        g = g[(g.close_h >= 24) & (g.open_h <= 22) & (g.reviews >= 30)].copy()   # open at 10pm and past midnight
        xy = np.c_[g.lat * 111, g.lon * 111 * np.cos(np.radians(g.lat.mean()))]  # km
        cent = xy[rng.choice(len(xy), 4, replace=False)]
        for _ in range(30):
            lab = ((xy[:, None] - cent[None]) ** 2).sum(-1).argmin(1)
            cent = np.array([xy[lab == k].mean(0) for k in range(4)])
        g["cluster"] = lab
        share = np.maximum(3, np.round(VENUES_PER_CITY * np.bincount(lab, minlength=4) / len(g))).astype(int)
        while share.sum() > VENUES_PER_CITY:
            share[share.argmax()] -= 1
        while share.sum() < VENUES_PER_CITY:
            share[share.argmin()] += 1
        # bars and clubs first, restaurants-with-a-bar only if a neighborhood runs short
        pick = pd.concat([g[g.cluster == k].sort_values(["restaurant", "reviews"], ascending=[True, False]).head(share[k])
                          for k in range(4)])
        names = [ZIP_NAMES.get(str(pick[pick.cluster == k].zip.mode()[0]), f"zone {k + 1}") for k in range(4)]
        top = [pick[pick.cluster == k].nlargest(1, "reviews").name.iloc[0] for k in range(4)]
        names = [n if names.index(n) == k else f"near {top[k]}" for k, n in enumerate(names)]   # keep names unique
        pick["hood"] = pick.cluster.map(dict(enumerate(names)))
        pick.drop(columns=["business_id"]).to_csv(f"data/yelp_{city.lower().replace(' ', '_')}_venues.csv", index=False)
        print(f"{city}: {len(bars[bars.city == city])} bars with Saturday hours, {len(g)} eligible, "
              f"18 picked across {pick.hood.unique().tolist()}")


if __name__ == "__main__":
    build()

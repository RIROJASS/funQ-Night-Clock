# irl.py - read back what groups value from what they did (inverse RL), per night clock
# (Miami weekend / weeknight from fieldwork, Philadelphia and New Orleans from Yelp, Madrid-style late start).
# Fits by maximum likelihood with the exact gradient through each group's lookahead.
# Also asks "do groups plan or react?" by comparing lookaheads of 0 (one-step logit), 1, 2 and the full night.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import city

# ---------------- settings you can turn ----------------
L2 = 0.001           # regularization: light pull toward 0 (0 = off, 0.01+ = strong at this data size)
LEARNING_RATE = 0.05
MAX_STEPS = 1500
CHECK_EVERY = 10     # early stopping: score the validation nights every 10 steps...
PATIENCE = 8         # ...and stop after 8 checks in a row with no improvement
LOOKAHEADS = {"react (logit)": 0, "1 move": 1, "2 moves": 2, "full night": city.H}

df = pd.read_csv("data/nights.csv")
rng = np.random.default_rng(0)
nights = df.night.unique()
test_n = set(rng.choice(nights, size=len(nights) // 5, replace=False))
val_n = set(rng.choice(sorted(set(nights) - test_n), size=len(nights) // 5, replace=False))
test, val = df[df.night.isin(test_n)], df[df.night.isin(val_n)]
train = df[~df.night.isin(test_n | val_n)]


def loglik(w, d, tables, horizon):
    """Average log-probability of the choices groups actually made, and its gradient."""
    Qs, dQs = city.q_values(w, tables, horizon)
    Q, dQ = np.stack(Qs), np.stack(dQs)
    i = (d.hour.values, d.stratum.values, d.location.values, d.energy.values)
    q, dq = Q[i], dQ[i]                                     # [decisions, actions], [decisions, actions, K]
    V = city.TEMPERATURE * np.logaddexp.reduce(q / city.TEMPERATURE, axis=1)
    pi = np.exp((q - V[:, None]) / city.TEMPERATURE)
    a = d.action.values
    ll = np.mean((q[np.arange(len(a)), a] - V) / city.TEMPERATURE)
    grad = (dq[np.arange(len(a)), a] - np.einsum("na,nak->nk", pi, dq)).mean(0) / city.TEMPERATURE
    return ll, grad


def fit(tables, tr, va, horizon):
    """Gradient ascent (Adam) with light L2 and early stopping on validation nights."""
    w, m, v = np.zeros(city.K), np.zeros(city.K), np.zeros(city.K)
    best, best_w, bad = -np.inf, w.copy(), 0
    for step in range(1, MAX_STEPS + 1):
        _, g = loglik(w, tr, tables, horizon)
        g -= L2 * w
        m, v = 0.9 * m + 0.1 * g, 0.999 * v + 0.001 * g ** 2
        w = w + LEARNING_RATE * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
        if step % CHECK_EVERY == 0:
            score = loglik(w, va, tables, horizon)[0]
            if score > best + 1e-5:
                best, best_w, bad = score, w.copy(), 0
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
    return best_w


fits, plan_rows, rec_rows = {}, [], []
for c in city.CONTEXTS:
    tables = city.build_tables(c)
    tr, va, te = (x[x.context == c] for x in (train, val, test))
    true = np.array(city.TRUE_WEIGHTS[c])
    fits[c] = {name: fit(tables, tr, va, hz) for name, hz in LOOKAHEADS.items()}
    fits[c]["tables"] = tables
    plan_rows.append([c] + [loglik(fits[c][n], te, tables, hz)[0] for n, hz in LOOKAHEADS.items()])
    w = fits[c]["2 moves"]
    top5 = np.abs(w[[0, 1, 2, 3, 5]])            # cover, guest list, table, DJ, door
    ranking_ok = top5[0] > top5[1] > max(top5[2], top5[3]) and min(top5[2], top5[3]) > top5[4]
    rec_rows.append([c, np.abs(w - true).mean(), np.corrcoef(w, true)[0, 1],
                     np.abs(fits[c]["react (logit)"] - true).mean(), ranking_ok])

rec = pd.DataFrame(rec_rows, columns=["context", "IRL avg error", "IRL corr", "logit avg error", "fieldwork ranking recovered"])
plan = pd.DataFrame(plan_rows, columns=["context"] + list(LOOKAHEADS))
print("Recovery (IRL assumes a 2-move lookahead):\n" + rec.round(3).to_string(index=False))
print("\nDo groups plan or react? Held-out log-likelihood (higher = better):\n" + plan.round(4).to_string(index=False))
pd.DataFrame([[c, f, city.TRUE_WEIGHTS[c][i], fits[c]["2 moves"][i], fits[c]["react (logit)"][i]]
              for c in city.CONTEXTS for i, f in enumerate(city.FEATURES)],
             columns=["context", "feature", "true", "irl", "logit"]).round(3).to_csv("data/recovered_weights.csv", index=False)
plan.round(4).to_csv("data/plan_or_react.csv", index=False)

# ---------------- charts (reference palette) ----------------
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#b3b2ab"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#ecebe6", "#fcfcfb"
PURPLE = "#8a5cd1"
COLOR = {"Miami weekend": BLUE, "Miami weeknight": ORANGE, "Philadelphia": AQUA, "New Orleans": PURPLE, "Late-start city": YELLOW}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#d6d5cf", "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "axes.spines.top": False, "axes.spines.right": False})
y = np.arange(city.K)

# 1) true vs recovered, one panel per night clock
fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True)
axes.flat[-1].axis("off")
for ax, c in zip(axes.flat, city.CONTEXTS):
    for off, vals, col, lab in [(-0.27, city.TRUE_WEIGHTS[c], GRAY, "true"), (0, fits[c]["2 moves"], BLUE, "inverse RL (2-move lookahead)"),
                                (0.27, fits[c]["react (logit)"], ORANGE, "one-step logit")]:
        ax.barh(y + off, vals, height=0.25, color=col, label=lab)
    ax.axvline(0, color="#d6d5cf", lw=1)
    ax.set_yticks(y, city.FEATURES)
    ax.invert_yaxis()
    ax.tick_params(labelbottom=True)
    ax.set_title(c, loc="left", color=INK, fontweight="bold")
    ax.grid(axis="x", color=GRID, lw=0.8)
fig.legend(*axes[0, 0].get_legend_handles_labels(), frameon=False, loc="upper right", ncol=3)
fig.supxlabel("weight (negative = avoids, positive = seeks)", color=MUTED)
fig.suptitle("Inverse RL reads back what each city's groups value (bar cities barely charge cover and New Orleans has no tables, so those weights are hard to read)",
             x=0.01, ha="left", color=INK, fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig("figures/1_true_vs_recovered.png", dpi=160)

# 2) headline: the pull toward high-energy spots by hour (recovered), and who is still out
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2))
hrs = np.arange(city.H)
for c in city.CONTEXTS:
    w = fits[c]["2 moves"]
    d = df[df.context == c]
    out = [(d[d.hour == h].action != city.HOME).sum() / d.night.nunique() for h in hrs]
    live = [h for h in hrs if out[h] > 0.03]
    pull = w[3] + w[4] * hrs / (city.H - 1)
    t = city.TRUE_WEIGHTS[c]
    a1.plot(live, pull[live], color=COLOR[c], lw=2.2)
    a1.plot(live, (t[3] + t[4] * hrs / (city.H - 1))[live], color=COLOR[c], lw=1.2, ls="--", alpha=0.7)
    a1.text(live[-1] + 0.15, pull[live[-1]], c, color=COLOR[c], va="center", fontsize=9)
    a2.plot(hrs, out, color=COLOR[c], lw=2.2)
    fall = max(h for h in hrs if out[h] >= 0.5)                     # label each curve where it starts to drop
    a2.text(fall + 0.15, out[fall], c, color=COLOR[c], ha="left", va="bottom", fontsize=9)
a1.set_title("Pull toward high-energy spots, by hour (solid = recovered, dashed = true)", loc="left", color=INK)
a1.set_ylabel("weight on DJ / vibe")
a2.set_title("Share of groups still out", loc="left", color=INK)
a2.set_ylim(0, 1.05)
for ax in (a1, a2):
    ax.set_xticks(hrs, city.HOURS)
    ax.set_xlim(-0.3, city.H + 1.2 if ax is a1 else city.H - 0.5)
    ax.grid(axis="y", color=GRID, lw=0.8)
fig.suptitle("Late clocks (New Orleans, Miami weekends) keep groups chasing energy deeper into the night; Philadelphia's 2am sends them home",
             x=0.01, ha="left", color=INK, fontsize=12)
fig.tight_layout()
fig.savefig("figures/2_night_clock_headline.png", dpi=160)

# 3) explain one Miami weekend night: why each move won
c = "Miami weekend"
w, (Phi, ok, ne) = fits[c]["2 moves"], fits[c]["tables"]
te = test[test.context == c]
night = te[te.type == "tourist"].groupby("night").filter(lambda g: len(g) >= 4 and g.action_name.nunique() >= 3).night.iloc[0]
steps = te[te.night == night]
r0 = steps.iloc[0]
fig, axes = plt.subplots(1, len(steps), figsize=(2.9 * len(steps), 4.6), sharex=True, sharey=True)
for ax, (_, r) in zip(axes, steps.iterrows()):
    contrib = Phi[r.stratum, r.hour, r.location, r.energy, r.action] * w
    ax.barh(y, contrib, color=[BLUE if v >= 0 else ORANGE for v in contrib], height=0.6)
    ax.axvline(0, color="#d6d5cf", lw=1)
    ax.set_yticks(y, city.FEATURES)
    ax.tick_params(labelbottom=True)
    ax.set_title(f"{city.HOURS[r.hour]}: {r.action_name}\nenergy {r.energy}", loc="left", color=INK, fontsize=9)
axes[0].invert_yaxis()
fig.supxlabel("contribution to the move's value (recovered weight x feature)", color=MUTED)
fig.suptitle(f"One Miami weekend night: a {r0.type} group of {r0.group_size} on a {r0.path} (night #{night})",
             x=0.01, ha="left", color=INK, fontsize=12)
fig.tight_layout()
fig.savefig("figures/3_explain_a_night.png", dpi=160)
# 4) the real night clocks: share of ALL bars open at each hour of a Saturday night, from Yelp
try:
    clk = pd.read_csv("data/yelp_open_by_hour.csv")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for c, col in [("Philadelphia", AQUA), ("New Orleans", PURPLE)]:
        d = clk[clk.city == c]
        ax.plot(d.hour, d.share_open, color=col, lw=2.4)
        ax.text(27.2, d[d.hour == 27].share_open.iloc[0] + 0.02, c, color=col, fontsize=10)
    ax.set_xticks(range(20, 32), ["8pm", "9pm", "10pm", "11pm", "12am", "1am", "2am", "3am", "4am", "5am", "6am", "7am"])
    ax.set_ylim(0, 0.9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_title("Share of bars open, Saturday night (Yelp Open Dataset)", loc="left", color=INK)
    fig.suptitle("Philadelphia's bars shut at 2am; New Orleans keeps a tail open until morning", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig("figures/4_real_closing_times.png", dpi=160)
except FileNotFoundError:
    print("no data/yelp_open_by_hour.csv - run python yelp.py to draw the real closing-time chart")
print("charts saved to figures/")

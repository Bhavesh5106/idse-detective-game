"""
🕵️ Detective P-Hacker
=====================
A single-file Streamlit game that teaches:

  1. Correlation vs. causation
  2. P-hacking (the multiple-comparisons problem)
  3. Confounding variables (and how "controlling for" them works)

Run with:
    pip install streamlit pandas numpy scipy
    streamlit run detective_p_hacker.py

Libraries used: streamlit, pandas, numpy, scipy.stats (plus the Python stdlib).
"""

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Detective P-Hacker", page_icon="🕵️", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# Game constants
# ─────────────────────────────────────────────────────────────────────────────
N_MONTHS = 36            # three years of monthly data
ALPHA = 0.05             # significance threshold
R_THRESHOLD = 0.8        # "shocking enough to print" threshold
POINTS_PUBLISH = 100     # circulation points for publishing a story
POINTS_CORRECT = 150     # integrity bonus for naming the true confounder
HINT_COST = 25           # cost of asking the intern for a lead

CONFOUNDERS = ["Summer Heat", "Economic Boom", "Exam Season"]

# Three engineered pairs. Each variable in a pair is built from the SAME hidden
# confounder plus independent noise, so they correlate strongly without either
# one causing the other.
PAIR_SPECS = [
    {
        "a": "Ice Cream Sales", "a_mean": 12000, "a_sd": 3500,
        "b": "Shark Attacks", "b_mean": 10, "b_sd": 3,
        "confounder": "Summer Heat", "loading": 0.94,
        "headline": "SHOCK STUDY: Ice Cream Is Turning Beachgoers Into Shark Bait!",
        "mechanism": (
            "Hot months send people to the ice-cream stand **and** into the ocean. "
            "More swimmers in the water means more shark encounters. Ice cream never "
            "touched a fin, because *Summer Heat* drives both series."
        ),
    },
    {
        "a": "Luxury Watch Sales", "a_mean": 850, "a_sd": 120,
        "b": "Stock Market Index", "b_mean": 4200, "b_sd": 350,
        "confounder": "Economic Boom", "loading": 0.93,
        "headline": "MARKET MYSTERY SOLVED: Luxury Watch Sales Control Wall Street!",
        "mechanism": (
            "A booming economy lifts stock prices **and** leaves consumers with spare "
            "cash for fancy watches. Neither causes the other, because the *Economic "
            "Boom* causes both."
        ),
    },
    {
        "a": "Library Late Fees", "a_mean": 2300, "a_sd": 600,
        "b": "Coffee Shop Revenue", "b_mean": 41000, "b_sd": 6500,
        "confounder": "Exam Season", "loading": 0.92,
        "headline": "EXCLUSIVE: Library Late Fees Are Secretly Fueling the Coffee Boom!",
        "mechanism": (
            "During *Exam Season* students swarm the library (and forget to return "
            "books) **and** pile into cafés for all-night study sessions. Late fees "
            "don't buy lattes; the exam calendar drives both."
        ),
    },
]

# 14 pure-noise variables (independent random draws).
NOISE_SPECS = [
    ("Cat Videos Watched", 5200, 900),
    ("Moustache Length", 14, 4),
    ("Pigeon Sightings", 310, 60),
    ("Lottery Tickets Sold", 78000, 9000),
    ("Podcast Downloads", 26000, 5000),
    ("Bus Delays", 420, 80),
    ("Parking Tickets Issued", 1900, 300),
    ("Garden Gnome Sales", 160, 35),
    ("Pizza Deliveries", 9800, 1400),
    ("Streetlight Outages", 55, 12),
    ("Yoga Class Attendance", 3400, 500),
    ("Sock Lint Collected", 22, 5),
    ("Crossword Puzzles Solved", 640, 110),
    ("Traffic Cone Thefts", 8, 3),
]

PAIR_LOOKUP = {frozenset((s["a"], s["b"])): s for s in PAIR_SPECS}


# ─────────────────────────────────────────────────────────────────────────────
# Data generation
# ─────────────────────────────────────────────────────────────────────────────
def _zscore(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / x.std(ddof=0)


def _make_confounders(rng, months):
    """Three hidden drivers, each z-scored."""
    n = len(months)

    # Summer Heat: yearly sine wave peaking in July.
    heat = np.sin(2 * np.pi * (months - 4) / 12) + 0.25 * rng.standard_normal(n)

    # Economic Boom: a smooth, persistent AR(1) drift.
    boom = np.zeros(n)
    boom[0] = rng.standard_normal()
    for t in range(1, n):
        boom[t] = 0.9 * boom[t - 1] + 0.5 * rng.standard_normal()

    # Exam Season: spikes in May/December, shoulders around them.
    exam = np.select(
        [np.isin(months, [5, 12]), np.isin(months, [1, 4, 6, 11])],
        [1.0, 0.45],
        default=0.0,
    ) + 0.15 * rng.standard_normal(n)

    return {
        "Summer Heat": _zscore(heat),
        "Economic Boom": _zscore(boom),
        "Exam Season": _zscore(exam),
    }


def _generate(seed):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=N_MONTHS, freq="MS")
    months = dates.month.to_numpy()
    hidden = _make_confounders(rng, months)

    data = {}
    for spec in PAIR_SPECS:
        load = spec["loading"]
        conf = hidden[spec["confounder"]]
        for side in ("a", "b"):
            z = load * conf + np.sqrt(1 - load**2) * rng.standard_normal(N_MONTHS)
            data[spec[side]] = np.round(spec[f"{side}_mean"] + spec[f"{side}_sd"] * z, 1)

    for name, mean, sd in NOISE_SPECS:
        data[name] = np.round(mean + sd * rng.standard_normal(N_MONTHS), 1)

    df = pd.DataFrame(data, index=dates)
    df = df[sorted(df.columns)]          # alphabetical, so column order gives nothing away
    df.index.name = "Month"
    hidden_df = pd.DataFrame(hidden, index=dates)
    return df, hidden_df


def _is_valid(df):
    """Engineered pairs must be strong; every other pair must be clearly weaker."""
    cols = list(df.columns)
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            r, p = stats.pearsonr(df[c1].to_numpy(), df[c2].to_numpy())
            if frozenset((c1, c2)) in PAIR_LOOKUP:
                if not (r > 0.82 and p < 0.01):
                    return False
            elif abs(r) >= 0.75:
                return False
    return True


@st.cache_data(show_spinner=False)
def build_dataset():
    """Deterministically search seeds until the dataset satisfies the game rules."""
    for seed in range(2026, 2026 + 2000):
        df, hidden = _generate(seed)
        if _is_valid(df):
            return df, hidden
    raise RuntimeError("Could not generate a valid dataset.")  # practically unreachable


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────
def pair_key(a, b):
    return tuple(sorted((a, b)))


def fmt_p(p):
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def correlation(df, a, b):
    r, p = stats.pearsonr(df[a].to_numpy(dtype=float), df[b].to_numpy(dtype=float))
    return float(r), float(p)


def partial_correlation(x, y, z):
    """Correlation of x and y after linearly removing z from both."""
    x, y, z = (np.asarray(v, dtype=float) for v in (x, y, z))
    design = np.column_stack([np.ones_like(z), z])
    rx = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    ry = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    r, _ = stats.pearsonr(rx, ry)
    r = float(r)
    dof = len(x) - 3
    denom = max(1 - r**2, 1e-12)
    t = r * np.sqrt(dof / denom)
    p = float(2 * stats.t.sf(abs(t), dof))
    return r, p


def all_pairs_table(df):
    rows = []
    cols = list(df.columns)
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            r, p = correlation(df, c1, c2)
            rows.append({
                "Variable 1": c1, "Variable 2": c2,
                "r": round(r, 3), "p-value": round(p, 5),
                "Engineered": frozenset((c1, c2)) in PAIR_LOOKUP,
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Session state + callbacks
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    if "initialized" in st.session_state:
        return
    st.session_state.update({
        "initialized": True,
        "phase": "hunt",          # hunt -> reality -> hunt ... -> finished
        "score": 0,
        "published": [],          # list of pair keys
        "tested": set(),          # unique pairs the player has looked at
        "story": None,
        "story_counter": 0,
        "checks_correct": 0,
        "checks_total": 0,
        "hints": [],
        "results": [],
        "celebrated": False,
    })


def reset_game():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    init_state()


def publish_story(a, b):
    """Callback for the 'Publish Story' button."""
    df, _ = build_dataset()
    key = pair_key(a, b)
    spec = PAIR_LOOKUP.get(frozenset(key))
    if spec is None or key in st.session_state.published or a == b:
        return
    r, p = correlation(df, a, b)
    if not (r > R_THRESHOLD and p < ALPHA):
        return

    others = [c for c in CONFOUNDERS if c != spec["confounder"]]
    options = [
        spec["confounder"],
        *others,
        f"Direct causation: {spec['a']} → {spec['b']}",
        "Pure coincidence (random chance)",
    ]
    order = np.random.default_rng().permutation(len(options))
    options = [options[i] for i in order]

    st.session_state.story_counter += 1
    st.session_state.story = {
        "id": st.session_state.story_counter,
        "key": key, "a": a, "b": b, "r": r, "p": p,
        "options": options, "answered": False, "warn": False,
        "choice": None, "correct": None,
    }
    st.session_state.score += POINTS_PUBLISH
    st.session_state.published.append(key)
    st.session_state.results.append({
        "Headline": spec["headline"], "Pair": f"{a} ↔ {b}",
        "Hidden confounder": spec["confounder"],
        "Raw r": round(r, 3), "r after controlling": None, "Fact-check": "—",
    })
    st.session_state.phase = "reality"


def lock_in_answer():
    """Callback for the 'Lock in answer' button."""
    story = st.session_state.story
    if story is None or story["answered"]:
        return
    choice = st.session_state.get(f"answer_{story['id']}")
    if choice is None:
        story["warn"] = True
        return
    df, hidden = build_dataset()
    spec = PAIR_LOOKUP[frozenset(story["key"])]
    correct = choice == spec["confounder"]
    story.update({"answered": True, "warn": False, "choice": choice, "correct": correct})
    st.session_state.checks_total += 1
    if correct:
        st.session_state.checks_correct += 1
        st.session_state.score += POINTS_CORRECT
    pr, _ = partial_correlation(df[story["a"]], df[story["b"]], hidden[spec["confounder"]])
    res = st.session_state.results[-1]
    res["r after controlling"] = round(pr, 3)
    res["Fact-check"] = "✅ Correct" if correct else "❌ Wrong"


def next_story():
    st.session_state.story = None
    if len(st.session_state.published) >= len(PAIR_SPECS):
        st.session_state.phase = "finished"
    else:
        st.session_state.phase = "hunt"


def buy_hint():
    """Reveal one variable from a still-unpublished engineered pair."""
    for spec in PAIR_SPECS:
        if pair_key(spec["a"], spec["b"]) in st.session_state.published:
            continue
        for side in ("a", "b"):
            if spec[side] not in st.session_state.hints:
                st.session_state.hints.append(spec[side])
                st.session_state.score = max(0, st.session_state.score - HINT_COST)
                return


# ─────────────────────────────────────────────────────────────────────────────
# UI pieces
# ─────────────────────────────────────────────────────────────────────────────
def newspaper_headline(text, kicker):
    html = (
        '<div style="border:3px double #888; padding:1rem 1.25rem; border-radius:6px; '
        'text-align:center; margin-bottom:0.75rem;">'
        '<div style="font-size:0.8rem; letter-spacing:0.25em; text-transform:uppercase; '
        f'opacity:0.7;">{kicker}</div>'
        '<div style="font-family:Georgia,\'Times New Roman\',serif; font-size:1.9rem; '
        f'font-weight:800; line-height:1.2;">{text}</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def zscore_frame(df, cols):
    return df[cols].apply(lambda s: (s - s.mean()) / s.std(ddof=0))


def render_sidebar():
    ss = st.session_state
    with st.sidebar:
        st.header("🗞️ The Daily Spurious")
        st.caption("Circulation desk · Editor-in-Chief: *you*")
        st.metric("Circulation score", ss.score)
        c1, c2 = st.columns(2)
        c1.metric("Published", f"{len(ss.published)}/{len(PAIR_SPECS)}")
        c2.metric("Fact-checks", f"{ss.checks_correct}/{ss.checks_total}")
        st.divider()
        with st.expander("🎓 Learning objectives", expanded=True):
            st.markdown(
                "**1 · P-hacking** *(Phase 1)*\n\n"
                "Testing many variable pairs until something is \"significant\" inflates "
                "false positives. Watch the multiple-comparisons meter.\n\n"
                "**2 · Correlation ≠ causation** *(Phase 2)*\n\n"
                "A tiny p-value says the association is unlikely to be chance. It says "
                "nothing about *why* the variables move together.\n\n"
                "**3 · Confounding variables** *(Phase 2)*\n\n"
                "A hidden third variable can drive both series. Controlling for it makes "
                "the \"relationship\" vanish."
            )
        with st.expander("📖 How to play"):
            st.markdown(
                f"1. Pick two variables and inspect the scatter plot.\n"
                f"2. Hunt for **r > {R_THRESHOLD}** *and* **p < {ALPHA}**.\n"
                f"3. Click **Publish Story** (+{POINTS_PUBLISH} pts).\n"
                f"4. Survive the **Reality Check**: name the hidden confounder "
                f"(+{POINTS_CORRECT} pts).\n"
                f"5. Publish all {len(PAIR_SPECS)} stories to win.\n\n"
                f"*Asking the intern for a lead costs {HINT_COST} pts.*"
            )
        st.button("🔄 Restart game", on_click=reset_game)


def render_hacking_meter(n_cols):
    """Multiple-comparisons tracker: the core p-hacking lesson."""
    n_tested = len(st.session_state.tested)
    total_pairs = n_cols * (n_cols - 1) // 2
    fwer = 1 - (1 - ALPHA) ** n_tested if n_tested else 0.0
    bonf = ALPHA / n_tested if n_tested else ALPHA

    with st.container(border=True):
        st.markdown("#### 🎰 The P-Hacking Meter")
        m1, m2, m3 = st.columns(3)
        m1.metric("Hypotheses tested", f"{n_tested} / {total_pairs}")
        m2.metric("Chance of ≥1 false alarm", f"{fwer:.0%}",
                  help="1 − (1 − α)^N if every pair were truly unrelated.")
        m3.metric("Bonferroni threshold", f"{bonf:.5f}",
                  help="α divided by the number of tests you've run.")
        st.progress(min(n_tested / total_pairs, 1.0))
        st.caption(
            f"🎓 **Data-science lens:** every pair you try is another lottery ticket. "
            f"Across all {total_pairs} possible pairs you'd expect roughly "
            f"{ALPHA * total_pairs:.0f} \"significant\" results at α = {ALPHA} even if "
            f"*nothing* were real. Honest researchers correct for this (e.g. Bonferroni); "
            f"corrupt editors keep clicking until something lights up."
        )


def render_hunt(df):
    st.subheader("Phase 1 · The P-Hack 🔍")
    st.markdown(
        f"Circulation is down. Dig through **{df.shape[1]} variables** × "
        f"**{df.shape[0]} months** of data and find a pair with "
        f"**r > {R_THRESHOLD}** and **p < {ALPHA}**. The facts matter less than the headline."
    )

    options = list(df.columns)
    c1, c2 = st.columns(2)
    x_var = c1.selectbox("Variable X", options, index=0, key="x_var")
    y_var = c2.selectbox("Variable Y", options, index=1, key="y_var")

    if x_var == y_var:
        st.info("Pick two *different* variables to compare.")
        render_hacking_meter(df.shape[1])
        return

    key = pair_key(x_var, y_var)
    st.session_state.tested.add(key)
    r, p = correlation(df, x_var, y_var)
    spec = PAIR_LOOKUP.get(frozenset(key))
    already = key in st.session_state.published
    eligible = (r > R_THRESHOLD) and (p < ALPHA) and (spec is not None) and not already

    left, right = st.columns([3, 2])

    with left:
        tab_scatter, tab_time = st.tabs(["📈 Scatter plot", "⏱️ Over time"])
        with tab_scatter:
            st.scatter_chart(df[[x_var, y_var]].reset_index(drop=True), x=x_var, y=y_var)
        with tab_time:
            st.line_chart(zscore_frame(df, [x_var, y_var]))
            st.caption("Both series are standardized (z-scores) so they share one axis. "
                       "A shared rhythm hints that something hidden may be driving both.")

    with right:
        with st.container(border=True):
            st.markdown("#### 🔬 Statistics")
            m1, m2 = st.columns(2)
            m1.metric("Pearson r", f"{r:+.3f}",
                      help="Strength and direction of the linear relationship (−1 to +1).")
            m2.metric("p-value", fmt_p(p),
                      help="How surprising an r this extreme would be if the variables were truly unrelated.")
            st.caption(f"n = {len(df)} monthly observations")

            n_tested = len(st.session_state.tested)
            survives = p < ALPHA / n_tested

            if already:
                st.info("📰 You already ran this story. Find a different pair!")
            elif eligible:
                st.success("🔥 Front-page material! Strong **and** significant.")
                if survives:
                    st.caption("🎓 It even survives a Bonferroni correction, so the "
                               "association is statistically real. *Real ≠ causal.*")
            elif r > R_THRESHOLD:
                st.warning("Strong, but not significant. Keep digging.")
            elif r < -R_THRESHOLD and p < ALPHA:
                st.warning("Strong but **negative**. The headline desk only buys "
                           "\"X goes up with Y\" stories.")
            elif p < ALPHA:
                st.warning(
                    "Statistically significant, but too weak to shock anyone "
                    f"(r ≤ {R_THRESHOLD}). 🎓 This is how p-hacking produces "
                    "false positives: with enough comparisons, noise crosses p < 0.05."
                )
            else:
                st.error("Not significant. This looks like noise.")

            st.button("📰 Publish Story", type="primary", disabled=not eligible,
                      on_click=publish_story, args=(x_var, y_var))

        with st.expander("🧑‍💼 Ask the intern for a lead"):
            st.caption(f"Costs {HINT_COST} points. Reveals one variable that has a "
                       "suspiciously strong partner somewhere in the data.")
            leads = [h for h in st.session_state.hints
                     if any(h in (s["a"], s["b"])
                            and pair_key(s["a"], s["b"]) not in st.session_state.published
                            for s in PAIR_SPECS)]
            remaining = any(
                s[side] not in st.session_state.hints
                for s in PAIR_SPECS
                if pair_key(s["a"], s["b"]) not in st.session_state.published
                for side in ("a", "b")
            )
            st.button("Get a lead", on_click=buy_hint, disabled=not remaining)
            if leads:
                st.markdown("**Leads:** " + ", ".join(f"`{h}`" for h in leads))

    render_hacking_meter(df.shape[1])


def render_reality(df, hidden):
    story = st.session_state.story
    spec = PAIR_LOOKUP[frozenset(story["key"])]
    a, b, conf = story["a"], story["b"], spec["confounder"]

    st.subheader("Phase 2 · The Reality Check 🧐")
    newspaper_headline(spec["headline"], "Late edition · Front page")
    st.caption(f"Your evidence: r = {story['r']:+.3f}, p = {fmt_p(story['p'])}. "
               f"+{POINTS_PUBLISH} circulation points banked.")

    if not story["answered"]:
        st.markdown(
            f"A rival statistician fires back: *\"{a} and {b} do move together, but one "
            f"did **not** cause the other.\"* Which hidden variable is really driving both?"
        )
        with st.expander("🔎 Re-examine the evidence", expanded=True):
            st.line_chart(zscore_frame(df, [a, b]))
            st.caption("Look at the shape of both series over three years. What real-world "
                       "pattern do they share?")
        st.radio("What is the true explanation?", story["options"], index=None,
                 key=f"answer_{story['id']}")
        st.button("🔒 Lock in answer", type="primary", on_click=lock_in_answer)
        if story["warn"]:
            st.warning("Pick an answer before locking in.")
        return

    # ── Answered: feedback and debrief ──
    choice = story["choice"]
    if story["correct"]:
        st.success(f"✅ Correct! The hidden confounder was **{conf}**. "
                   f"+{POINTS_CORRECT} integrity points.")
    else:
        st.error(f"❌ Not quite. You chose *{choice}*, but the real driver was **{conf}**.")
        if choice.startswith("Direct causation"):
            st.caption("🎓 This is the classic error: correlation does not imply causation.")
        elif choice.startswith("Pure coincidence"):
            st.caption(f"🎓 With p = {fmt_p(story['p'])}, chance is a poor explanation. The "
                       "association is real, but its *source* is a third variable.")
        else:
            st.caption("🎓 That's a different hidden variable in this dataset; it doesn't "
                       "track the shared pattern in both series.")

    st.markdown(f"**What really happened:** {spec['mechanism']}")

    pr, ppr = partial_correlation(df[a], df[b], hidden[conf])
    c1, c2, c3 = st.columns(3)
    c1.metric("Raw correlation", f"{story['r']:+.3f}")
    c2.metric(f"Controlling for {conf}", f"{pr:+.3f}")
    c3.metric("p-value after controlling", fmt_p(ppr))
    st.caption("🎓 **Controlling for a confounder** means removing its linear influence from "
               "both variables (via regression residuals) and re-checking the correlation. "
               "When the link collapses, the confounder was doing the work.")

    chart_df = zscore_frame(df, [a, b]).copy()
    chart_df[f"{conf} (hidden)"] = hidden[conf].to_numpy()
    st.line_chart(chart_df)
    st.caption("The hidden variable (third line) tracks both series. This is your smoking gun.")

    with st.container(border=True):
        st.markdown(
            "**🎓 Concepts you just practiced**\n\n"
            "- **Spurious correlation:** a strong association with no direct causal link.\n"
            "- **Confounding variable:** a third factor that influences both variables.\n"
            "- **Why p < 0.05 isn't enough:** significance measures surprise under \"no "
            "relationship\", not the presence of a causal mechanism.\n"
            "- **Fix:** controlled experiments, randomization, or statistical adjustment "
            "for known confounders."
        )

    done = len(st.session_state.published) >= len(PAIR_SPECS)
    st.button("📊 See the final report" if done else "🔙 Back to the newsroom",
              type="primary", on_click=next_story)


def render_finished(df):
    ss = st.session_state
    if not ss.celebrated:
        st.balloons()
        ss.celebrated = True

    correct, total = ss.checks_correct, ss.checks_total
    if correct == total:
        title = "🏅 Reformed Detective: you'd make a fine data scientist"
    elif correct >= 2:
        title = "🧐 Skeptical Reporter: mostly immune to spurious stories"
    else:
        title = "📰 Hopeless Hack: the confounders got you"

    st.subheader("Final Report 📊")
    st.success(f"All {len(PAIR_SPECS)} stories published! Final score: **{ss.score}**  ·  {title}")

    results = pd.DataFrame(ss.results)
    st.dataframe(results, hide_index=True)

    st.markdown("#### 🎰 The Dredge: what a brute-force search would have found")
    tbl = all_pairs_table(df)
    sig = tbl[tbl["p-value"] < ALPHA]
    false_pos = sig[~sig["Engineered"]]
    total = len(tbl)
    st.markdown(
        f"Out of **{total}** possible pairs, **{len(sig)}** had p < {ALPHA}. "
        f"Only **{len(PAIR_SPECS)}** were engineered, so **{len(false_pos)}** were "
        f"**pure-noise false positives**, close to the ~{ALPHA * (total - len(PAIR_SPECS)):.0f} "
        f"expected by chance."
    )
    if len(false_pos):
        top = false_pos.assign(abs_r=false_pos["r"].abs()).sort_values(
            "abs_r", ascending=False).drop(columns=["abs_r", "Engineered"]).head(8)
        st.caption("Strongest noise-only \"discoveries\". Each is statistically significant "
                   "yet meaningless:")
        st.dataframe(top, hide_index=True)

    with st.container(border=True):
        st.markdown(
            "**🎓 What an honest scientist does differently**\n\n"
            "1. **Pre-register hypotheses** before looking at the data.\n"
            "2. **Correct for multiple comparisons** (Bonferroni, Benjamini–Hochberg).\n"
            "3. **Hunt for confounders** and adjust for them.\n"
            "4. **Prefer experiments** (randomization breaks confounding).\n"
            "5. **Report every test you ran**, not just the winners."
        )
    st.button("🔁 Play again", type="primary", on_click=reset_game)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    init_state()
    df, hidden = build_dataset()
    render_sidebar()

    st.title("🕵️ Detective P-Hacker")
    st.caption("You're the corrupt editor of *The Daily Spurious*. Find a statistic so "
               "shocking that readers can't look away. Truth optional.")

    phase = st.session_state.phase
    if phase == "hunt":
        render_hunt(df)
    elif phase == "reality" and st.session_state.story is not None:
        render_reality(df, hidden)
    elif phase == "finished":
        render_finished(df)
    else:  # defensive fallback
        st.session_state.phase = "hunt"
        render_hunt(df)


main()
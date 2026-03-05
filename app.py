# app.py
import math
import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

# ============================================================
# Helpers: odds + formatting
# ============================================================
def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal odds."""
    if odds == 0:
        raise ValueError("American odds cannot be 0.")
    if odds > 0:
        return 1.0 + (odds / 100.0)
    return 1.0 + (100.0 / abs(odds))


def american_to_implied_prob(odds: float) -> float:
    """Convert American odds to implied probability (includes vig)."""
    if odds == 0:
        raise ValueError("American odds cannot be 0.")
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def prob_to_decimal_odds(p: float) -> float:
    """Decimal odds from probability (fair odds, no vig)."""
    if p <= 0 or p >= 1:
        raise ValueError("Probability must be between 0 and 1 (exclusive).")
    return 1.0 / p


def clamp(p: float, lo: float = 1e-9, hi: float = 1 - 1e-9) -> float:
    return float(max(lo, min(hi, p)))


def pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def money(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.2f}"


# ============================================================
# Markov Chain: directional decay parlay
# States:
#   Transient: k = 0..5  (have cleared k legs; about to attempt leg k+1)
#   Absorbing: Bust, Jackpot
# ============================================================
def build_transition_matrix(p_legs: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """
    p_legs: shape (6,), probability of winning each leg
    Returns:
      P: (8x8) full transition matrix
      state_names
    """
    if p_legs.shape != (6,):
        raise ValueError("p_legs must be shape (6,)")

    n_transient = 6
    n_absorb = 2
    n = n_transient + n_absorb

    BUST = n_transient
    JACKPOT = n_transient + 1

    state_names = (
        ["Start (0 legs cleared)"]
        + [f"Cleared {k} leg(s)" for k in range(1, 6)]
        + ["Bust (Absorbing)", "Jackpot (Absorbing)"]
    )

    P = np.zeros((n, n), dtype=float)

    for k in range(n_transient):
        p_win = float(p_legs[k])
        p_lose = 1.0 - p_win

        if k < n_transient - 1:
            P[k, k + 1] = p_win
            P[k, BUST] = p_lose
        else:
            P[k, JACKPOT] = p_win
            P[k, BUST] = p_lose

    P[BUST, BUST] = 1.0
    P[JACKPOT, JACKPOT] = 1.0

    return P, state_names


def partition_Q_R(P: np.ndarray, n_transient: int = 6) -> tuple[np.ndarray, np.ndarray]:
    Q = P[:n_transient, :n_transient]
    R = P[:n_transient, n_transient:]
    return Q, R


def fundamental_matrix(Q: np.ndarray) -> np.ndarray:
    I = np.eye(Q.shape[0])
    return np.linalg.inv(I - Q)


def analytical_outputs(p_legs: np.ndarray) -> dict:
    P, state_names = build_transition_matrix(p_legs)
    Q, R = partition_Q_R(P, n_transient=6)
    F = fundamental_matrix(Q)
    B = F @ R

    p_jackpot = float(B[0, 1])
    p_bust = 1.0 - p_jackpot

    ones = np.ones((F.shape[0], 1))
    t = (F @ ones).flatten()
    expected_legs_attempted = float(t[0])

    return {
        "P": P,
        "Q": Q,
        "R": R,
        "F": F,
        "B": B,
        "state_names": state_names,
        "p_jackpot": p_jackpot,
        "p_bust": p_bust,
        "expected_legs_attempted": expected_legs_attempted,
    }


def simulate_parlay(p_legs: np.ndarray, stake: float, net_payout: float, trials: int, seed: int | None) -> dict:
    rng = np.random.default_rng(seed)

    jackpots = 0
    legs_attempted_list = []
    profit_list = []

    for _ in range(trials):
        legs_attempted = 0
        alive = True

        for i in range(6):
            legs_attempted += 1
            if rng.random() <= p_legs[i]:
                continue
            alive = False
            break

        if alive:
            jackpots += 1
            profit = net_payout
        else:
            profit = -stake

        legs_attempted_list.append(legs_attempted)
        profit_list.append(profit)

    jackpots_rate = jackpots / trials
    avg_legs = float(np.mean(legs_attempted_list))
    avg_profit = float(np.mean(profit_list))

    return {
        "jackpot_rate": jackpots_rate,
        "avg_legs": avg_legs,
        "avg_profit": avg_profit,
        "legs_attempted": legs_attempted_list,
        "profits": profit_list,
    }


# ============================================================
# De-vig / "casino fee" adjustment (single-side approximation)
# ============================================================
def remove_margin_single_side(implied_p: float, margin: float) -> float:
    """
    Approximation when you only have ONE side's odds:

    If sportsbook inflates probabilities by ~ (1 + margin),
    then implied_p ≈ fair_p * (1 + margin)
    => fair_p ≈ implied_p / (1 + margin)

    This is not exact without both sides of the market.
    """
    fair = implied_p / (1.0 + margin)
    return clamp(fair)


# ============================================================
# Online insights (simple + straightforward)
# ============================================================
@st.cache_data(ttl=10 * 60, show_spinner=False)
def fetch_google_news_rss(query: str, max_items: int = 8) -> list[dict]:
    """
    Pull headlines from Google News RSS (no API key).
    """
    if not query.strip():
        return []

    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()

    root = ET.fromstring(r.text)
    channel = root.find("channel")
    if channel is None:
        return []

    items = []
    for item in channel.findall("item")[:max_items]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        source = ""
        source_el = item.find("source")
        if source_el is not None and source_el.text:
            source = source_el.text.strip()

        if title and link:
            items.append({"title": title, "link": link, "pubDate": pub, "source": source})
    return items


@st.cache_data(ttl=5 * 60, show_spinner=False)
def fetch_odds_api_events(sport_key: str, regions: str, markets: str, odds_format: str, api_key: str) -> list[dict]:
    """
    The Odds API v4: get events + odds. Requires api_key.
    """
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
    }
    r = requests.get(url, params=params, timeout=12)
    r.raise_for_status()
    return r.json()


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="6-Leg Parlay Lab", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.1rem; padding-bottom: 2.2rem; }
      div[data-testid="stMetricValue"] { font-size: 1.55rem; }
      .small-note { font-size: 0.92rem; opacity: 0.85; }
      .card {
        border: 1px solid rgba(49, 51, 63, 0.18);
        border-radius: 14px;
        padding: 14px 14px 10px 14px;
        background: rgba(255,255,255,0.02);
      }
      .tight hr { margin: 0.7rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("6-Leg Parlay Lab")
st.caption("Markov chain (analytical) + Monte Carlo (simulation) + optional market/news pull-ins.")

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to",
        ["Build Parlay", "Insights (Online)", "Diagnostics"],
        index=0,
    )

    st.divider()
    st.subheader("Model settings")
    dev_mode = st.toggle("Developer mode", value=False, help="Show matrices + deeper reconciliation.")
    st.caption("Tip: keep Dev mode off for a cleaner UI.")

# -----------------------------
# Shared inputs (parlay builder)
# -----------------------------
def parlay_input_panel():
    st.subheader("Inputs")

    left, right = st.columns([0.62, 0.38], gap="large")

    with left:
        st.markdown("#### Bet legs (6)")
        names, odds = [], []

        for i in range(6):
            with st.container(border=True):
                c1, c2 = st.columns([0.64, 0.36], gap="medium")
                with c1:
                    nm = st.text_input(f"Leg {i+1} name", value=f"Leg {i+1}", key=f"name_{i}")
                with c2:
                    od = st.number_input(
                        f"Leg {i+1} American odds",
                        value=-110,
                        step=1,
                        key=f"odds_{i}",
                        help="Examples: -110, +150, -200",
                    )
                names.append(nm.strip() if nm else f"Leg {i+1}")
                odds.append(float(od))

    with right:
        st.markdown("#### Stake + simulation")
        stake = st.number_input("Stake ($)", min_value=0.01, value=10.00, step=1.00)
        trials = st.number_input("Simulation trials", min_value=1000, max_value=750000, value=50000, step=1000)
        seed = st.number_input("Random seed (0 = random)", min_value=0, value=0, step=1)

        st.markdown("#### Casino vig / margin (estimate)")
        preset = st.selectbox(
            "Preset",
            [
                "Custom",
                "Standard -110 style pricing (approx 4.5% overround on a 50/50 two-way line)",
                "Higher-margin / parlay-heavy environment (example 8%)",
                "Very high margin (example 10%)",
            ],
            index=0,
        )

        preset_map = {
            "Custom": None,
            "Standard -110 style pricing (approx 4.5% overround on a 50/50 two-way line)": 0.0454,
            "Higher-margin / parlay-heavy environment (example 8%)": 0.08,
            "Very high margin (example 10%)": 0.10,
        }

        if preset_map[preset] is None:
            margin_pct = st.number_input(
                "Margin / vig (%)",
                min_value=0.0,
                max_value=25.0,
                value=5.0,
                step=0.1,
                help="This is an *estimate* of the embedded sportsbook edge.",
            )
            margin = float(margin_pct) / 100.0
        else:
            margin = float(preset_map[preset])
            st.caption(f"Using preset margin: **{margin*100:.2f}%**")

        st.markdown('<div class="small-note">De-vig note: exact true odds require both sides of a market. With only one side, we apply a transparent approximation: fair_p ≈ implied_p / (1+margin).</div>', unsafe_allow_html=True)

    return names, odds, float(stake), int(trials), int(seed), float(margin)


def compute_all(names, odds, stake, trials, seed, margin):
    # Compute implied probabilities + decimals
    implied_probs = []
    decimal_odds = []
    input_errors = []

    for i in range(6):
        try:
            ip = american_to_implied_prob(odds[i])
            dec = american_to_decimal(odds[i])
            implied_probs.append(float(ip))
            decimal_odds.append(float(dec))
        except Exception as e:
            input_errors.append(f"Leg {i+1}: {e}")
            implied_probs.append(float("nan"))
            decimal_odds.append(float("nan"))

    if input_errors:
        st.error("Fix these input issues:\n- " + "\n- ".join(input_errors))
        st.stop()

    p_offered = np.array(implied_probs, dtype=float)

    # Offered parlay payout math from odds
    parlay_decimal_offered = float(np.prod(decimal_odds))
    net_payout_offered = stake * (parlay_decimal_offered - 1.0)

    # De-vig (estimated) "true" probabilities & odds
    p_true = np.array([remove_margin_single_side(p, margin) for p in p_offered], dtype=float)
    true_decimal_legs = np.array([prob_to_decimal_odds(p) for p in p_true], dtype=float)
    parlay_decimal_true = float(np.prod(true_decimal_legs))
    net_payout_true = stake * (parlay_decimal_true - 1.0)

    # Analytical (Markov chain) offered
    ana_offered = analytical_outputs(p_legs=p_offered)
    p_jackpot_offered = float(ana_offered["p_jackpot"])
    expected_legs_attempted = float(ana_offered["expected_legs_attempted"])

    # Direct product check (should match jackpot for directional-decay parlay)
    p_jackpot_product = float(np.prod(p_offered))
    analytical_diff = abs(p_jackpot_offered - p_jackpot_product)

    ev_profit_offered = p_jackpot_offered * net_payout_offered - (1.0 - p_jackpot_offered) * stake

    # Analytical (Markov chain) true
    ana_true = analytical_outputs(p_legs=p_true)
    p_jackpot_true = float(ana_true["p_jackpot"])
    ev_profit_true = p_jackpot_true * net_payout_true - (1.0 - p_jackpot_true) * stake

    # Simulation offered
    sim_seed = None if seed == 0 else seed
    sim_offered = simulate_parlay(
        p_legs=p_offered,
        stake=stake,
        net_payout=net_payout_offered,
        trials=trials,
        seed=sim_seed,
    )

    # Simulation true (optional / slower). Keep it aligned with what "true payout" would do.
    sim_true = simulate_parlay(
        p_legs=p_true,
        stake=stake,
        net_payout=net_payout_true,
        trials=trials,
        seed=sim_seed,
    )

    return {
        "p_offered": p_offered,
        "p_true": p_true,
        "decimal_odds": np.array(decimal_odds, dtype=float),
        "true_decimal_legs": true_decimal_legs,
        "parlay_decimal_offered": parlay_decimal_offered,
        "net_payout_offered": net_payout_offered,
        "parlay_decimal_true": parlay_decimal_true,
        "net_payout_true": net_payout_true,
        "ana_offered": ana_offered,
        "ana_true": ana_true,
        "p_jackpot_offered": p_jackpot_offered,
        "p_jackpot_true": p_jackpot_true,
        "expected_legs_attempted": expected_legs_attempted,
        "p_jackpot_product": p_jackpot_product,
        "analytical_diff": analytical_diff,
        "ev_profit_offered": ev_profit_offered,
        "ev_profit_true": ev_profit_true,
        "sim_offered": sim_offered,
        "sim_true": sim_true,
    }


# ============================================================
# Page: Build Parlay
# ============================================================
if page == "Build Parlay":
    names, odds, stake, trials, seed, margin = parlay_input_panel()
    results = compute_all(names, odds, stake, trials, seed, margin)

    st.divider()

    # --- Leg table (offered + estimated true) ---
    st.subheader("Leg breakdown")

    legs_df = pd.DataFrame(
        {
            "Leg": [f"{i+1}" for i in range(6)],
            "Bet name": names,
            "American odds (offered)": odds,
            "Implied P(win) (offered)": [float(p) for p in results["p_offered"]],
            "Implied P(win) %": [pct(p) for p in results["p_offered"]],
            "Decimal odds (offered)": [round(d, 4) for d in results["decimal_odds"]],
            "P(win) (de-vig est.)": [float(p) for p in results["p_true"]],
            "P(win) % (de-vig est.)": [pct(p) for p in results["p_true"]],
            "Decimal odds (de-vig est.)": [round(d, 4) for d in results["true_decimal_legs"]],
        }
    )
    st.dataframe(legs_df, use_container_width=True, hide_index=True)

    # --- Summary metrics ---
    st.subheader("Results (Offered vs De-vig Estimate)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Parlay decimal (offered)", f'{results["parlay_decimal_offered"]:.4f}')
    c2.metric("Net payout if win (offered)", money(results["net_payout_offered"]))
    c3.metric("Jackpot P (analytical, offered)", pct(results["p_jackpot_offered"]))
    c4.metric("Jackpot rate (sim, offered)", pct(results["sim_offered"]["jackpot_rate"]))

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Parlay decimal (de-vig est.)", f'{results["parlay_decimal_true"]:.4f}')
    d2.metric("Net payout if win (de-vig est.)", money(results["net_payout_true"]))
    d3.metric("Jackpot P (analytical, de-vig est.)", pct(results["p_jackpot_true"]))
    d4.metric("Jackpot rate (sim, de-vig est.)", pct(results["sim_true"]["jackpot_rate"]))

    st.divider()

    # --- EV panel ---
    st.subheader("Expected value")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Expected legs attempted (analytical)", f'{results["expected_legs_attempted"]:.3f}')
    e2.metric("Avg legs attempted (sim, offered)", f'{results["sim_offered"]["avg_legs"]:.3f}')
    e3.metric("Expected profit/bet (offered)", money(results["ev_profit_offered"]))
    e4.metric("Expected profit/bet (de-vig est.)", money(results["ev_profit_true"]))

    st.markdown("### Expected amount of money a player would make")
    st.info(
        f"- **Analytical expected profit per bet (offered):** {money(results['ev_profit_offered'])}\n"
        f"- **Analytical expected profit per bet (de-vig est.):** {money(results['ev_profit_true'])}\n"
        f"- **Simulation avg profit per bet (offered):** {money(results['sim_offered']['avg_profit'])}\n"
        f"- **Simulation avg profit per bet (de-vig est.):** {money(results['sim_true']['avg_profit'])}\n"
        f"- If you repeat the same parlay **N** times, a rough expectation is **N × expected profit per bet**."
    )

    # --- Survival-to-leg probabilities (intuitive) ---
    survive_to = [1.0]
    prod = 1.0
    for i in range(5):
        prod *= results["p_offered"][i]
        survive_to.append(prod)

    survival_df = pd.DataFrame(
        {
            "Milestone": [f"Reach leg {k+1}" for k in range(6)],
            "Probability (offered)": survive_to,
            "Probability % (offered)": [pct(x) for x in survive_to],
        }
    )
    st.markdown("#### Probability of reaching each leg (analytical, offered)")
    st.dataframe(survival_df, use_container_width=True, hide_index=True)

    # --- Accuracy checks ---
    st.markdown("#### Accuracy checks")
    tol = 1e-12
    if results["analytical_diff"] <= tol:
        st.success(f"Analytical jackpot probability matches direct product within tolerance (diff={results['analytical_diff']:.2e}).")
    else:
        st.warning(
            f"Analytical jackpot probability differs from direct product (diff={results['analytical_diff']:.2e}). "
            "This should be ~0 for a pure directional-decay parlay; if not, review inputs / numeric stability."
        )

    st.caption(
        "Reminder: de-vig is an estimate with one-sided odds. For exact no-vig odds, you need both sides of the market (so you can divide by the overround)."
    )

    # --- Developer diagnostics (matrices + distributions) ---
    if dev_mode:
        st.divider()
        st.subheader("Dev Mode: Matrices + reconciliation")

        P = results["ana_offered"]["P"]
        Q = results["ana_offered"]["Q"]
        R = results["ana_offered"]["R"]
        F = results["ana_offered"]["F"]
        B = results["ana_offered"]["B"]
        state_names = results["ana_offered"]["state_names"]

        P_df = pd.DataFrame(P, index=state_names, columns=state_names)
        Q_df = pd.DataFrame(Q, index=state_names[:6], columns=state_names[:6])
        R_df = pd.DataFrame(R, index=state_names[:6], columns=["Bust", "Jackpot"])
        F_df = pd.DataFrame(F, index=state_names[:6], columns=state_names[:6])
        B_df = pd.DataFrame(B, index=state_names[:6], columns=["P(Bust)", "P(Jackpot)"])

        tabs = st.tabs(["P", "Q & R", "Fundamental F", "Absorption B", "Simulation summaries"])

        with tabs[0]:
            st.dataframe(P_df, use_container_width=True)

        with tabs[1]:
            a, b = st.columns(2)
            with a:
                st.dataframe(Q_df, use_container_width=True)
            with b:
                st.dataframe(R_df, use_container_width=True)

        with tabs[2]:
            st.dataframe(F_df, use_container_width=True)

        with tabs[3]:
            st.dataframe(B_df, use_container_width=True)

        with tabs[4]:
            legs_series = pd.Series(results["sim_offered"]["legs_attempted"], name="legs_attempted")
            prof_series = pd.Series(results["sim_offered"]["profits"], name="profit")

            a, b = st.columns(2)
            with a:
                st.write(legs_series.describe())
            with b:
                st.write(prof_series.describe())

# ============================================================
# Page: Insights (Online)
# ============================================================
elif page == "Insights (Online)":
    st.subheader("Insights (Online)")
    st.caption("This panel pulls public headlines and (optionally) market odds data. It does not guarantee outcomes.")

    st.markdown("#### Describe the event / market")
    q1, q2 = st.columns([0.62, 0.38], gap="large")

    with q1:
        keywords = st.text_input(
            "Search keywords (teams, players, event, league)",
            value="",
            placeholder="e.g., Lakers vs Warriors spread injury report sharp money",
        )

    with q2:
        st.markdown("**Optional: Odds API market pull**")
        # Prefer st.secrets, fallback to env var
        odds_api_key = ""
        try:
            odds_api_key = st.secrets.get("ODDS_API_KEY", "")
        except Exception:
            odds_api_key = os.getenv("ODDS_API_KEY", "")

        has_key = bool(odds_api_key)
        st.caption("Set ODDS_API_KEY in Streamlit secrets or env var.")

    st.divider()

    # --- News RSS (no key) ---
    st.markdown("### What’s being said (headlines)")
    if not keywords.strip():
        st.info("Enter keywords above to fetch headlines.")
    else:
        with st.spinner("Fetching headlines..."):
            try:
                items = fetch_google_news_rss(keywords, max_items=10)
            except Exception as e:
                items = []
                st.error(f"News fetch failed: {e}")

        if not items:
            st.warning("No headlines returned. Try different keywords.")
        else:
            for it in items:
                src = f" — {it['source']}" if it.get("source") else ""
                st.markdown(f"- [{it['title']}]({it['link']}){src}")
                if it.get("pubDate"):
                    st.caption(it["pubDate"])

    st.divider()

    # --- Odds API (requires key) ---
    st.markdown("### Market odds snapshot (optional)")
    if not has_key:
        st.info("Add an ODDS_API_KEY to enable market odds pulls (The Odds API v4).")
        st.caption("Docs: /v4/sports/{sport_key}/odds with regions/markets/oddsFormat.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            sport_key = st.text_input("Sport key", value="americanfootball_nfl", help="Example: americanfootball_nfl")
        with c2:
            regions = st.selectbox("Regions", ["us", "us2", "eu", "uk", "au"], index=0)
        with c3:
            markets = st.selectbox("Markets", ["h2h", "spreads", "totals"], index=0)
        with c4:
            odds_format = st.selectbox("Odds format", ["american", "decimal"], index=0)

        if st.button("Fetch odds"):
            with st.spinner("Fetching odds from The Odds API..."):
                try:
                    data = fetch_odds_api_events(
                        sport_key=sport_key.strip(),
                        regions=regions,
                        markets=markets,
                        odds_format=odds_format,
                        api_key=odds_api_key,
                    )
                    st.success(f"Fetched {len(data)} event(s). Showing a compact view (first 10).")
                except Exception as e:
                    data = []
                    st.error(f"Odds API fetch failed: {e}")

            if data:
                # Compact view: event + first bookmaker + first market
                rows = []
                for ev in data[:10]:
                    home = ev.get("home_team", "")
                    away = ev.get("away_team", "")
                    commence = ev.get("commence_time", "")
                    books = ev.get("bookmakers", []) or []
                    if not books:
                        rows.append({"Event": f"{away} @ {home}", "Start": commence, "Book": "", "Market": "", "Outcomes": ""})
                        continue
                    book = books[0]
                    mks = book.get("markets", []) or []
                    if not mks:
                        rows.append({"Event": f"{away} @ {home}", "Start": commence, "Book": book.get("title",""), "Market": "", "Outcomes": ""})
                        continue
                    mk = mks[0]
                    outs = mk.get("outcomes", []) or []
                    out_str = "; ".join([f"{o.get('name','')}: {o.get('price','')}" for o in outs[:3]])
                    rows.append(
                        {
                            "Event": f"{away} @ {home}",
                            "Start": commence,
                            "Book": book.get("title", ""),
                            "Market": mk.get("key", ""),
                            "Outcomes": out_str,
                        }
                    )

                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.caption("Use this as *context* (injuries, line movement, consensus pricing), not as a promise of profitability.")

# ============================================================
# Page: Diagnostics
# ============================================================
else:
    st.subheader("Diagnostics")
    st.markdown(
        """
        This page is for sanity-checking assumptions and making the model harder to break.

        **Key facts for your current structure (directional-decay parlay):**
        - Analytical jackpot probability should equal **product of per-leg win probabilities**.
        - Markov chain method is an exact re-expression of that process; simulation should converge to it with enough trials.
        """
    )

    st.markdown("#### Quick numeric stress test")
    p = st.slider("Example constant leg win probability", min_value=0.05, max_value=0.95, value=0.52, step=0.01)
    stake = st.number_input("Stake ($)", min_value=0.01, value=10.00, step=1.00)
    parlay_decimal = st.number_input("Parlay decimal odds (assumed)", min_value=1.01, value=50.0, step=0.5)
    net_payout = stake * (parlay_decimal - 1.0)

    p_legs = np.array([p] * 6, dtype=float)
    ana = analytical_outputs(p_legs)
    p_j = float(ana["p_jackpot"])
    p_prod = float(np.prod(p_legs))
    ev = p_j * net_payout - (1 - p_j) * stake

    a, b, c = st.columns(3)
    a.metric("Analytical P(jackpot)", pct(p_j))
    b.metric("Product P(jackpot)", pct(p_prod))
    c.metric("Expected profit/bet", money(ev))

    st.write({"abs_diff": abs(p_j - p_prod)})

st.caption("Educational tool only — do not treat as financial advice.")

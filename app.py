# app.py
import math
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Helpers: odds + formatting
# -----------------------------
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


def pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def money(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.2f}"


# -----------------------------
# Markov Chain: directional decay parlay
# States:
#   Transient: k = 0..5  (have cleared k legs; about to attempt leg k+1)
#   Absorbing: Bust, Jackpot
# -----------------------------
def build_transition_matrix(p_legs: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """
    p_legs: shape (6,), probability of winning each leg
    Returns:
      P: (8x8) full transition matrix
      state_names
    """
    if p_legs.shape != (6,):
        raise ValueError("p_legs must be shape (6,)")

    # 6 transient states + 2 absorbing
    n_transient = 6
    n_absorb = 2
    n = n_transient + n_absorb

    # Indexing
    BUST = n_transient
    JACKPOT = n_transient + 1

    state_names = (
        ["Start (0 legs cleared)"]
        + [f"Cleared {k} leg(s)" for k in range(1, 6)]
        + ["Bust (Absorbing)", "Jackpot (Absorbing)"]
    )

    P = np.zeros((n, n), dtype=float)

    # Transient transitions
    for k in range(n_transient):
        p_win = float(p_legs[k])
        p_lose = 1.0 - p_win

        if k < n_transient - 1:
            # win -> next transient (k+1), lose -> bust
            P[k, k + 1] = p_win
            P[k, BUST] = p_lose
        else:
            # final leg attempt: win -> jackpot, lose -> bust
            P[k, JACKPOT] = p_win
            P[k, BUST] = p_lose

    # Absorbing states self-loop
    P[BUST, BUST] = 1.0
    P[JACKPOT, JACKPOT] = 1.0

    return P, state_names


def partition_Q_R(P: np.ndarray, n_transient: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Partition P into Q (transient->transient) and R (transient->absorbing)."""
    Q = P[:n_transient, :n_transient]
    R = P[:n_transient, n_transient:]
    return Q, R


def fundamental_matrix(Q: np.ndarray) -> np.ndarray:
    """F = (I - Q)^(-1)"""
    I = np.eye(Q.shape[0])
    return np.linalg.inv(I - Q)


def analytical_outputs(p_legs: np.ndarray, stake: float) -> dict:
    """
    Compute:
      - P matrix, Q, R, F
      - jackpot probability from start
      - expected legs attempted (expected life)
      - theoretical expected profit per bet
      - survival probabilities to each leg
    """
    P, state_names = build_transition_matrix(p_legs)
    Q, R = partition_Q_R(P, n_transient=6)
    F = fundamental_matrix(Q)
    B = F @ R  # absorption probabilities from each transient state

    # Absorbing columns: [Bust, Jackpot]
    p_jackpot = float(B[0, 1])
    p_bust = 1.0 - p_jackpot

    # Expected number of steps until absorption: t = F * 1
    ones = np.ones((F.shape[0], 1))
    t = (F @ ones).flatten()
    expected_legs_attempted = float(t[0])

    # Parlay net payout based on odds product (computed outside usually),
    # but EV formula uses p_jackpot and net payout.
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
                # win leg, continue
                continue
            else:
                # bust
                alive = False
                break

        if alive:
            jackpots += 1
            profit = net_payout  # net profit on a win
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


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="6-Leg Parlay Validator", layout="wide")

# Minimal styling
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; }
      div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

top_left, top_right = st.columns([0.8, 0.2])
with top_left:
    st.title("6-Leg Parlay Validator")
    st.caption(
        "Analytical Markov Chain + Monte Carlo simulation (Directional Decay: progress forward or bust)."
    )
with top_right:
    dev_mode = st.toggle("Dev mode", value=False, help="Show full matrices, intermediate outputs, and diagnostics.")

st.divider()

# --- Inputs (ONLY what you requested, plus sim trials) ---
st.subheader("Inputs")

colA, colB = st.columns([0.55, 0.45], gap="large")

with colA:
    st.markdown("#### Bet legs (6)")
    names = []
    odds = []

    for i in range(6):
        c1, c2 = st.columns([0.62, 0.38])
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

with colB:
    st.markdown("#### Stake + simulation")
    stake = st.number_input("Stake ($)", min_value=0.01, value=10.00, step=1.00)
    trials = st.number_input("Simulation trials", min_value=1000, max_value=500000, value=25000, step=1000)
    seed = st.number_input("Random seed (optional)", min_value=0, value=0, step=1, help="Set 0 for random each run.")

# Compute implied probabilities + decimals
implied_probs = []
decimal_odds = []
input_errors = []

for i in range(6):
    try:
        ip = american_to_implied_prob(odds[i])
        dec = american_to_decimal(odds[i])
        implied_probs.append(ip)
        decimal_odds.append(dec)
    except Exception as e:
        input_errors.append(f"Leg {i+1}: {e}")
        implied_probs.append(float("nan"))
        decimal_odds.append(float("nan"))

if input_errors:
    st.error("Fix these input issues:\n- " + "\n- ".join(input_errors))
    st.stop()

p_legs = np.array(implied_probs, dtype=float)

# Parlay payout math from odds
parlay_decimal = float(np.prod(decimal_odds))
net_payout = stake * (parlay_decimal - 1.0)  # profit if win

# Show implied prob per bet (REQUIRED)
st.markdown("#### Implied probability for each leg (from American odds)")
legs_df = pd.DataFrame(
    {
        "Leg": [f"{i+1}" for i in range(6)],
        "Bet name": names,
        "American odds": odds,
        "Implied P(win)": [float(p) for p in implied_probs],
        "Implied P(win) %": [pct(p) for p in implied_probs],
        "Decimal odds": [round(d, 4) for d in decimal_odds],
    }
)
st.dataframe(legs_df, use_container_width=True, hide_index=True)

# --- Run engines ---
# Analytical (Markov chain)
ana = analytical_outputs(p_legs=p_legs, stake=float(stake))

p_jackpot = ana["p_jackpot"]
expected_legs_attempted = ana["expected_legs_attempted"]

theoretical_expected_profit = p_jackpot * net_payout - (1.0 - p_jackpot) * stake
theoretical_expected_end_value = stake + theoretical_expected_profit  # "stake returned" notionally

# Simulation
sim_seed = None if int(seed) == 0 else int(seed)
sim = simulate_parlay(
    p_legs=p_legs,
    stake=float(stake),
    net_payout=float(net_payout),
    trials=int(trials),
    seed=sim_seed,
)

# --- Outputs ---
st.subheader("Results")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Parlay decimal odds", f"{parlay_decimal:.4f}")
m2.metric("Net payout if win", money(net_payout))
m3.metric("Jackpot probability (Analytical)", pct(p_jackpot))
m4.metric("Jackpot rate (Simulation)", pct(sim["jackpot_rate"]))

m5, m6, m7, m8 = st.columns(4)
m5.metric("Expected legs attempted (Analytical)", f"{expected_legs_attempted:.3f}")
m6.metric("Avg legs attempted (Simulation)", f"{sim['avg_legs']:.3f}")
m7.metric("Expected profit per bet (Analytical)", money(theoretical_expected_profit))
m8.metric("Avg profit per bet (Simulation)", money(sim["avg_profit"]))

# REQUIRED: "expected amount of money a player would make"
st.markdown("### Expected amount of money a player would make")
st.info(
    f"- **Analytical expected profit per bet:** {money(theoretical_expected_profit)}\n"
    f"- **Simulation average profit per bet:** {money(sim['avg_profit'])}\n"
    f"- **If you repeated this same parlay N times**, a rough expectation is **N × expected profit per bet**."
)

# Optional: survival-to-leg probabilities (nice + intuitive; also matches packet outputs)
# "survive to attempt leg k" = product of p_1..p_{k-1}  (for k=1 survival=1)
survive_to = [1.0]
prod = 1.0
for i in range(5):  # survive to attempt legs 2..6
    prod *= p_legs[i]
    survive_to.append(prod)

survival_df = pd.DataFrame(
    {
        "Milestone": [f"Reach leg {k+1}" for k in range(6)],
        "Probability": survive_to,
        "Probability %": [pct(x) for x in survive_to],
    }
)
st.markdown("#### Probability of reaching each leg (analytical)")
st.dataframe(survival_df, use_container_width=True, hide_index=True)

# --- Dev Mode Diagnostics ---
if dev_mode:
    st.divider()
    st.subheader("Dev Mode: Matrices + reconciliation")

    st.caption(
        "Packet requirements: isolate Q/R, compute fundamental matrix F=(I−Q)^−1, absorption B=F×R, and reconcile with simulation."
    )
    st.caption("Reference: dual-engine validator + required outputs/processing. " + ":contentReference[oaicite:2]{index=2}")

    P = ana["P"]
    Q = ana["Q"]
    R = ana["R"]
    F = ana["F"]
    B = ana["B"]
    state_names = ana["state_names"]

    # Label matrices
    P_df = pd.DataFrame(P, index=state_names, columns=state_names)
    Q_df = pd.DataFrame(Q, index=state_names[:6], columns=state_names[:6])
    R_df = pd.DataFrame(R, index=state_names[:6], columns=["Bust", "Jackpot"])
    F_df = pd.DataFrame(F, index=state_names[:6], columns=state_names[:6])
    B_df = pd.DataFrame(B, index=state_names[:6], columns=["P(Bust)", "P(Jackpot)"])

    tabs = st.tabs(["P", "Q & R", "Fundamental F", "Absorption B", "Simulation Distributions"])

    with tabs[0]:
        st.markdown("**Full transition matrix (P)**")
        st.dataframe(P_df, use_container_width=True)

    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Q (Transient → Transient)**")
            st.dataframe(Q_df, use_container_width=True)
        with c2:
            st.markdown("**R (Transient → Absorbing)**")
            st.dataframe(R_df, use_container_width=True)

    with tabs[2]:
        st.markdown("**Fundamental matrix F = (I − Q)⁻¹**")
        st.dataframe(F_df, use_container_width=True)

    with tabs[3]:
        st.markdown("**Absorption probabilities B = F × R**")
        st.dataframe(B_df, use_container_width=True)
        st.caption(
            "From Start, Jackpot probability is B[Start, Jackpot]. Expected Value follows P(Jackpot) × NetPayout − P(Bust) × Stake."
        )

    with tabs[4]:
        # Show simple hist-style summaries without forcing matplotlib colors
        legs_series = pd.Series(sim["legs_attempted"], name="legs_attempted")
        prof_series = pd.Series(sim["profits"], name="profit")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Legs attempted (simulation) - summary**")
            st.write(legs_series.describe())
        with c2:
            st.markdown("**Profit per bet (simulation) - summary**")
            st.write(prof_series.describe())

st.caption(
    "This app implements the packet’s directional-decay parlay as a Markov chain (transient legs with a bust exit, plus a jackpot absorbing state) and validates via simulation."
    + " :contentReference[oaicite:3]{index=3}"
)

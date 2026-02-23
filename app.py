# app.py
# Streamlit app: Parlay State Space Explorer (Directional Decay)
# - Supports 2–6 legs (you can "try fewer legs")
# - Supports American odds (+500 style)
# - Shows probability, payout, EV, break-even, and intuitive improvement tips
#
# Run: streamlit run app.py

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st

# Optional PDF text extraction (best effort)
try:
    import PyPDF2  # type: ignore
    PYPDF2_AVAILABLE = True
except Exception:
    PYPDF2_AVAILABLE = False


# ----------------------------
# Odds / payout helpers
# ----------------------------
def american_to_decimal(american: float) -> float:
    """Convert American odds to decimal odds (includes stake)."""
    if american == 0:
        raise ValueError("American odds cannot be 0.")
    if american > 0:
        return 1.0 + (american / 100.0)
    return 1.0 + (100.0 / abs(american))


def decimal_to_american(decimal_odds: float) -> float:
    """Convert decimal odds to American odds (approx)."""
    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must be > 1.0")
    if decimal_odds >= 2.0:
        return (decimal_odds - 1.0) * 100.0
    return -100.0 / (decimal_odds - 1.0)


def implied_prob_from_american(american: float) -> float:
    """American odds implied probability (no vig removal)."""
    if american == 0:
        raise ValueError("American odds cannot be 0.")
    if american > 0:
        return 100.0 / (american + 100.0)
    return abs(american) / (abs(american) + 100.0)


def profit_and_return(stake: float, american_odds: float) -> Tuple[float, float]:
    """Return (profit_if_win, total_return_if_win)."""
    dec = american_to_decimal(american_odds)
    total_return = stake * dec
    profit = total_return - stake
    return profit, total_return


def format_american(odds: float) -> str:
    if math.isinf(odds):
        return "∞"
    sign = "+" if odds >= 0 else ""
    return f"{sign}{odds:.0f}"


# ----------------------------
# Parlay math (N legs)
# ----------------------------
@dataclass
class ParlayMetrics:
    legs: int
    jackpot_prob: float
    bust_prob: float
    expected_legs_won: float
    expected_steps_to_absorption: float
    fair_decimal_odds: float
    fair_american_odds: float
    expected_value: float
    expected_profit: float
    breakeven_jackpot_prob: float
    roi: float
    median_outcome: str
    risk_of_ruin: float


def jackpot_probability(ps: List[float]) -> float:
    jp = 1.0
    for p in ps:
        jp *= p
    return jp


def expected_legs_won(ps: List[float]) -> float:
    """
    E[wins] = sum_{k=1..N} prod_{i=1..k} p_i
    """
    cum = 1.0
    total = 0.0
    for p in ps:
        cum *= p
        total += cum
    return total


def build_transition_matrix(ps: List[float]) -> np.ndarray:
    """
    Build transition matrix for N legs with directional decay.

    States:
      S0..S(N-1) = alive after k wins (k=0..N-1)
      BUST, JACKPOT absorbing

    Total states = (N) transient + 2 absorbing = N+2
    """
    n = len(ps)
    if n < 2 or n > 6:
        raise ValueError("This app supports 2–6 legs.")

    size = n + 2
    bust_idx = n
    jack_idx = n + 1

    P = np.zeros((size, size), dtype=float)

    # Transient progression
    for k in range(1, n):  # legs 1..(n-1): S(k-1)->S(k)
        p = ps[k - 1]
        P[k - 1, k] = p
        P[k - 1, bust_idx] = 1.0 - p

    # Final transient: S(n-1) -> JACKPOT or BUST
    pn = ps[-1]
    P[n - 1, jack_idx] = pn
    P[n - 1, bust_idx] = 1.0 - pn

    # Absorbing
    P[bust_idx, bust_idx] = 1.0
    P[jack_idx, jack_idx] = 1.0

    return P


def absorbing_chain_expected_steps(P: np.ndarray, n_transient: int) -> float:
    """
    Expected steps to absorption from S0 using absorbing Markov chain formula.
    """
    Q = P[:n_transient, :n_transient]
    I = np.eye(n_transient)
    F = np.linalg.inv(I - Q)
    t = F @ np.ones((n_transient, 1))
    return float(t[0, 0])


def compute_metrics(ps: List[float], stake: float, parlay_american_odds: float) -> ParlayMetrics:
    n = len(ps)
    jp = jackpot_probability(ps)
    bust = 1.0 - jp

    exp_wins = expected_legs_won(ps)

    Pmat = build_transition_matrix(ps)
    exp_steps = absorbing_chain_expected_steps(Pmat, n_transient=n)

    # Payout side (standard: lose => $0, win => stake * decimal)
    profit_if_win, total_return_if_win = profit_and_return(stake, parlay_american_odds)
    EV = jp * total_return_if_win
    exp_profit = EV - stake
    roi = exp_profit / stake if stake > 0 else float("nan")

    # Fair odds implied by your modeled jackpot probability (no vig)
    if jp > 0:
        fair_dec = 1.0 / jp
        fair_am = decimal_to_american(fair_dec)
    else:
        fair_dec = float("inf")
        fair_am = float("inf")

    dec = american_to_decimal(parlay_american_odds)
    breakeven_jp = 1.0 / dec

    median_outcome = "BUST" if jp < 0.5 else "JACKPOT"
    risk_of_ruin = bust  # one bet

    return ParlayMetrics(
        legs=n,
        jackpot_prob=jp,
        bust_prob=bust,
        expected_legs_won=exp_wins,
        expected_steps_to_absorption=exp_steps,
        fair_decimal_odds=fair_dec,
        fair_american_odds=fair_am,
        expected_value=EV,
        expected_profit=exp_profit,
        breakeven_jackpot_prob=breakeven_jp,
        roi=roi,
        median_outcome=median_outcome,
        risk_of_ruin=risk_of_ruin,
    )


# ----------------------------
# Diagrams (ASCII + Graphviz DOT)
# ----------------------------
def make_state_names(n: int) -> Tuple[List[str], str, str]:
    transient = [f"S{k}" for k in range(n)]
    bust = "BUST"
    jack = "JACKPOT"
    return transient, bust, jack


def ascii_diagram(ps: List[float], highlight: Optional[str] = None) -> str:
    n = len(ps)
    transient, bust, jack = make_state_names(n)

    def H(name: str) -> str:
        return f"[{name}]" if highlight == name else name

    q = [1.0 - p for p in ps]
    fp = [f"{p:.3f}" for p in ps]
    fq = [f"{x:.3f}" for x in q]

    # Build top chain line
    # S0 -> S1 -> ... -> S(n-1) -> JACKPOT
    chain_parts = []
    for i in range(n - 1):
        chain_parts.append(f"{H(transient[i])} ──win p{i+1}={fp[i]}──▶ ")
    chain_parts.append(f"{H(transient[n-1])} ──win p{n}={fp[n-1]}──▶ {H(jack)}")
    chain_line = "".join(chain_parts)

    # Bust arrows (simple)
    bust_lines = []
    for i in range(n):
        bust_lines.append(f"{H(transient[i])} ──lose q{i+1}={fq[i]}──▶ {H(bust)}")

    return (
        f"{n}-Leg Parlay State Space (Directional Decay)\n\n"
        "Legend:\n"
        "  p_i  = P(win leg i)     q_i = 1 - p_i = P(lose leg i)\n"
        "  S_k  = alive after k legs cleared (k wins in a row)\n"
        "  BUST and JACKPOT are absorbing\n\n"
        f"{chain_line}\n\n"
        "Directional decay exits (any loss => BUST):\n"
        + "\n".join(bust_lines)
        + "\n\nConstraints:\n"
        "- Only forward transitions among S-states (no backward edges, no resets).\n"
        "- Any loss transitions to BUST immediately.\n"
        f"- Winning leg {n} transitions to JACKPOT.\n"
    )


def dot_graph(ps: List[float], highlight: Optional[str], show_bust_edges: bool) -> str:
    n = len(ps)
    transient, bust, jack = make_state_names(n)

    def node_style(name: str) -> str:
        if name in (bust, jack):
            base = 'shape=doublecircle'
        else:
            base = 'shape=box'
        if highlight == name:
            base += ', style="filled", fillcolor="lightyellow"'
        return base

    lines = [
        "digraph G {",
        'rankdir="LR";',
        'labelloc="t";',
        f'label="{n}-Leg Parlay State Space (Directional Decay)";',
        'node [fontname="Helvetica"];',
        'edge [fontname="Helvetica"];'
    ]

    # Nodes
    for s in transient + [bust, jack]:
        lines.append(f'{s} [{node_style(s)}];')

    # Forward win edges
    for i in range(n - 1):
        lines.append(f'{transient[i]} -> {transient[i+1]} [label="win p{i+1}={ps[i]:.3f}"];')
    lines.append(f'{transient[n-1]} -> {jack} [label="win p{n}={ps[n-1]:.3f}"];')

    # Bust edges
    if show_bust_edges:
        for i in range(n):
            lines.append(f'{transient[i]} -> {bust} [label="lose q{i+1}={1.0-ps[i]:.3f}"];')

    # Absorbing self loops
    lines.append(f'{bust} -> {bust} [label="1", color="gray"];')
    lines.append(f'{jack} -> {jack} [label="1", color="gray"];')

    lines.append(f'{{rank=same; {bust}; {jack};}}')
    lines.append("}")
    return "\n".join(lines)


# ----------------------------
# Intuitive improvement helpers
# ----------------------------
def delta_needed_to_hit_target(jp_current: float, jp_target: float, ps: List[float]) -> Optional[float]:
    """
    Rough “if all legs were increased equally” estimate:
    Find delta added to each p_i (clipped to 1) that achieves jp_target.
    """
    if jp_current <= 0:
        return None
    if jp_current >= jp_target:
        return 0.0

    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        prod = 1.0
        for p in ps:
            prod *= min(1.0, p + mid)
        if prod >= jp_target:
            hi = mid
        else:
            lo = mid
    return hi


def leg_sensitivity(ps: List[float], bump: float = 0.01) -> List[Tuple[int, float, float]]:
    """
    For JP = Π p_i, estimate gain in JP if one leg is improved by +bump (absolute).
    Returns list of (leg_index_1based, jp_gain, p_current), sorted desc by gain.
    """
    base = jackpot_probability(ps)
    out = []
    for i in range(len(ps)):
        ps2 = ps.copy()
        ps2[i] = min(1.0, ps2[i] + bump)
        gain = jackpot_probability(ps2) - base
        out.append((i + 1, gain, ps[i]))
    return sorted(out, key=lambda x: x[1], reverse=True)


# ----------------------------
# PDF text extraction (optional)
# ----------------------------
def extract_pdf_text(file) -> str:
    if not PYPDF2_AVAILABLE:
        return "PyPDF2 is not installed in this environment."
    try:
        reader = PyPDF2.PdfReader(file)
        pages = []
        for i, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            pages.append(f"--- Page {i+1} ---\n{txt}")
        out = "\n\n".join(pages).strip()
        return out if out else "No extractable text found (PDF may be scanned or image-based)."
    except Exception as e:
        return f"Could not extract text from PDF. Error: {e}"


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Parlay State Space Explorer (+American Odds)", layout="wide")

st.title("Parlay State Space Explorer (Directional Decay) — with +500 Style Odds")
st.write(
    "Pick **2–6 legs**, adjust each leg’s win probability, enter parlay **American odds** (e.g., +500), "
    "and see probability + payout + EV + simple guidance for improving the ticket.\n\n"
    "**Educational tool only — not betting advice.**"
)

with st.sidebar:
    st.header("Inputs")

    # New feature: choose number of legs
    n_legs = st.slider("Number of legs", min_value=2, max_value=6, value=6, step=1)

    st.subheader("Per-leg win probabilities")
    st.caption("These are your assumptions. Smaller changes compound across legs.")
    default_p = 0.55
    ps_all = []
    for i in range(6):
        ps_all.append(
            st.slider(
                f"p{i+1} (win prob leg {i+1})",
                min_value=0.0,
                max_value=1.0,
                value=default_p,
                step=0.01
            )
        )
    # Use only the first n legs
    ps = ps_all[:n_legs]

    st.divider()
    st.subheader("Parlay odds (+500 style)")
    parlay_odds = st.number_input(
        "American odds for the parlay",
        value=500,
        step=10,
        help="Examples: +500, +1200, -110. (0 is not allowed.)"
    )
    if parlay_odds == 0:
        st.error("American odds cannot be 0. Please change it.")

    stake = st.number_input(
        "Stake ($)",
        min_value=0.0,
        value=100.0,
        step=10.0,
        help="How much you bet up front."
    )

    st.divider()
    st.subheader("Visualization")
    highlight = st.selectbox(
        "Highlight a state",
        options=["(none)"] + [f"S{k}" for k in range(n_legs)] + ["BUST", "JACKPOT"],
        index=0
    )
    highlight_state = None if highlight == "(none)" else highlight
    show_bust_edges = st.checkbox("Show BUST transitions", value=True)

    st.divider()
    st.subheader("Optional: Load Problem Set PDF")
    uploaded_pdf = st.file_uploader("Upload a PDF (optional)", type=["pdf"])

if parlay_odds == 0:
    st.stop()

# Compute core objects
P = build_transition_matrix(ps)
metrics = compute_metrics(ps, stake, parlay_odds)

diagram_txt = ascii_diagram(ps, highlight=highlight_state)
dot = dot_graph(ps, highlight=highlight_state, show_bust_edges=show_bust_edges)

profit_if_win, total_return_if_win = profit_and_return(stake, parlay_odds)
dec_odds = american_to_decimal(parlay_odds)
implied_prob = implied_prob_from_american(parlay_odds)

jp = metrics.jackpot_prob
breakeven = metrics.breakeven_jackpot_prob
gap = breakeven - jp
delta_equal = delta_needed_to_hit_target(jp, breakeven, ps) if breakeven <= 1.0 else None

# New feature: compare fewer legs (2..n) with same odds (or "fair odds" view)
# Note: In reality odds would change as legs change; we show probability-side + what *fair odds* would be.
comparisons = []
for k in range(2, n_legs + 1):
    ps_k = ps_all[:k]
    m_k = compute_metrics(ps_k, stake, parlay_odds)
    comparisons.append(m_k)

tab1, tab2, tab3, tab4 = st.tabs(["State Space", "Odds & Money", "Try Fewer Legs (Compare)", "Problem Set (optional)"])

with tab1:
    colA, colB = st.columns([1, 1])

    with colA:
        st.subheader("ASCII Diagram")
        st.code(diagram_txt, language="text")

        with st.expander("State definitions (quick reference)", expanded=True):
            st.markdown(
                f"""
- **S0**: Start / 0 legs cleared (alive)  
- **S1..S{n_legs-1}**: Alive after that many legs cleared  
- **BUST**: terminal failure (absorbing)  
- **JACKPOT**: terminal success (absorbing)
                """.strip()
            )

    with colB:
        st.subheader("Graph View")
        st.graphviz_chart(dot, use_container_width=True)

        with st.expander("Directional decay (plain-English)", expanded=True):
            st.markdown(
                """
You only move **forward** if you keep winning.  
At every step there’s a chance you **drop out** into **BUST**, and there’s no coming back.  
That repeated “drop out” chance is why hit probability **decays** as you add legs.
                """.strip()
            )

with tab2:
    st.subheader("Odds & Money (Non-math view)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Parlay Odds", format_american(parlay_odds))
    c2.metric("Decimal Odds", f"{dec_odds:.3f}x return")
    c3.metric("Stake", f"${stake:,.2f}")
    c4.metric("Profit if Win", f"${profit_if_win:,.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Total Return if Win", f"${total_return_if_win:,.2f}")
    c6.metric("Your Model: Jackpot Prob", f"{jp:.6f}")
    c7.metric("Market Implied Prob (from odds)", f"{implied_prob:.6f}")
    c8.metric("Break-even Jackpot Prob", f"{breakeven:.6f}")

    st.divider()
    st.subheader("Expected outcome (based on your probabilities + odds)")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Expected Value (EV)", f"${metrics.expected_value:,.2f}")
    d2.metric("Expected Profit", f"${metrics.expected_profit:,.2f}")
    d3.metric("ROI (Expected Profit / Stake)", f"{metrics.roi*100:.2f}%")
    d4.metric("Chance you lose the stake", f"{metrics.risk_of_ruin*100:.2f}%")

    st.caption(
        "Standard parlay payout assumption: lose → $0 return; win all legs → stake × decimal odds. "
        "This app uses that standardized structure."
    )

    with st.expander("Transition matrix (technical)", expanded=False):
        st.caption("State order: S0..S(n-1), BUST, JACKPOT")
        st.dataframe(P, use_container_width=True)

with tab3:
    st.subheader("Try Fewer Legs (Compare 2 → N)")
    st.caption(
        "This compares **hit probability** and **fair odds** as you reduce legs, "
        "using your first k leg probabilities. In real sportsbooks, the offered odds would change too."
    )

    # Table-like display
    header_cols = st.columns([1, 1, 1, 1, 1])
    header_cols[0].markdown("**Legs**")
    header_cols[1].markdown("**Hit Prob (JP)**")
    header_cols[2].markdown("**Bust Prob**")
    header_cols[3].markdown("**Fair Odds (no vig)**")
    header_cols[4].markdown("**Break-even JP @ your odds**")

    for m in comparisons:
        row = st.columns([1, 1, 1, 1, 1])
        row[0].write(f"{m.legs}")
        row[1].write(f"{m.jackpot_prob:.6f}")
        row[2].write(f"{m.bust_prob:.6f}")
        row[3].write(f"{format_american(m.fair_american_odds)}  (dec {m.fair_decimal_odds:.2f}x)")
        row[4].write(f"{m.breakeven_jackpot_prob:.6f}")

    st.divider()
    st.subheader("Simple takeaways")
    st.markdown(
        """
- Dropping a leg usually **increases hit probability a lot** (because you remove one multiplication by p).
- “Fair odds” are what the payout would be if it exactly matched your modeled probability (no house edge).
- If you want **more chance to win**, fewer legs is the cleanest lever.
        """.strip()
    )

    st.divider()
    st.subheader("Which leg should you improve?")
    impacts = leg_sensitivity(ps, bump=0.01)
    st.write("**If you improve a single leg by +1% (absolute), these are the JP gains:**")
    for leg, gain, pcur in impacts:
        st.write(f"- Leg {leg}: p={pcur:.2f} → +0.01 gives **+{gain:.6f}** JP increase")

    st.divider()
    st.subheader("Break-even guidance")
    if gap <= 0:
        st.success(
            f"Your modeled hit probability ({jp:.6f}) is **at or above** break-even ({breakeven:.6f}) "
            f"for odds {format_american(parlay_odds)}."
        )
    else:
        st.warning(
            f"To break even at odds {format_american(parlay_odds)}, you need hit probability **≥ {breakeven:.6f}**. "
            f"Right now you have **{jp:.6f}** (short by {gap:.6f})."
        )
        if delta_equal is not None:
            st.info(
                f"Rule-of-thumb: if every leg improved equally, you’d need about **+{delta_equal:.2%}** added to each p "
                f"(clipped at 1.0) to reach break-even."
            )

with tab4:
    st.subheader("Problem Set PDF (optional)")
    if uploaded_pdf is None:
        st.info("Upload the PDF in the sidebar to view extracted text here.")
    else:
        st.caption("Best-effort text extraction (scanned PDFs may not extract well without OCR).")
        st.text_area("Extracted text", value=extract_pdf_text(uploaded_pdf), height=450)

st.caption("Run locally with: `streamlit run app.py`")

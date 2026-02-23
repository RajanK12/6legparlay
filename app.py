import streamlit as st
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# Optional PDF text extraction (best effort, no OCR)
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except Exception:
    PYPDF2_AVAILABLE = False


# ----------------------------
# Core model utilities
# ----------------------------
STATE_NAMES = ["S0", "S1", "S2", "S3", "S4", "S5", "BUST", "JACKPOT"]
TRANSIENT = ["S0", "S1", "S2", "S3", "S4", "S5"]
ABSORBING = ["BUST", "JACKPOT"]


@dataclass
class ParlayMetrics:
    jackpot_prob: float
    bust_prob: float
    expected_legs_survived: float
    expected_steps_to_absorption: float


def build_transition_matrix(ps: List[float]) -> np.ndarray:
    """
    Build 8x8 transition matrix P for states:
    S0,S1,S2,S3,S4,S5,BUST,JACKPOT

    Directional decay:
      S(k-1) -> S(k) with p_k
      S(k-1) -> BUST with 1-p_k
      S5 -> JACKPOT with p6
      BUST and JACKPOT are absorbing
    """
    if len(ps) != 6:
        raise ValueError("Expected 6 probabilities p1..p6")

    P = np.zeros((8, 8), dtype=float)

    # Transient chain
    for k in range(1, 6):  # k=1..5 transitions S(k-1)->S(k)
        p = ps[k - 1]
        P[k - 1, k] = p
        P[k - 1, 6] = 1.0 - p  # to BUST

    # Last transient: S5 -> JACKPOT or BUST
    p6 = ps[5]
    P[5, 7] = p6
    P[5, 6] = 1.0 - p6

    # Absorbing
    P[6, 6] = 1.0  # BUST
    P[7, 7] = 1.0  # JACKPOT

    return P


def absorbing_chain_metrics(P: np.ndarray) -> ParlayMetrics:
    """
    Compute absorption probabilities and expected time to absorption from S0.
    Uses standard absorbing Markov chain formulas:
      Partition P = [[Q, R],
                     [0, I]]
      Fundamental matrix F = (I - Q)^(-1)
      Absorption probs B = F R
      Expected steps to absorption t = F 1
    Interpret "expected legs survived" as expected number of wins before bust/jackpot,
    which equals expected transient steps that advance along the chain (wins),
    BUT simpler for this structure:
      expected legs survived = sum_{k=1..6} P(reach attempt for leg k) * P(win leg k | reach)
                           = sum_{k=1..6} prod_{i=1..k} p_i
    We compute it directly for clarity + numerical stability.
    """
    # Transient indices 0..5, absorbing 6..7
    Q = P[:6, :6]
    R = P[:6, 6:]

    I = np.eye(Q.shape[0])
    # Fundamental matrix
    F = np.linalg.inv(I - Q)

    # Absorption probabilities from each transient start state
    B = F @ R  # shape (6,2): columns correspond to [BUST, JACKPOT]
    bust_prob = float(B[0, 0])
    jackpot_prob = float(B[0, 1])

    # Expected steps to absorption from each transient state
    ones = np.ones((6, 1))
    t = F @ ones
    expected_steps_to_absorption = float(t[0, 0])

    # Expected legs survived (wins before termination) for this chain
    # E[wins] = sum_{k=1..6} prod_{i=1..k} p_i
    # (because you must win first k legs to count the k-th win)
    ps = [
        float(P[0, 1]),  # p1
        float(P[1, 2]),  # p2
        float(P[2, 3]),  # p3
        float(P[3, 4]),  # p4
        float(P[4, 5]),  # p5
        float(P[5, 7]),  # p6
    ]
    cumprod = 1.0
    expected_legs_survived = 0.0
    for p in ps:
        cumprod *= p
        expected_legs_survived += cumprod

    return ParlayMetrics(
        jackpot_prob=jackpot_prob,
        bust_prob=bust_prob,
        expected_legs_survived=expected_legs_survived,
        expected_steps_to_absorption=expected_steps_to_absorption,
    )


def ascii_diagram(ps: List[float], highlight: Optional[str] = None) -> str:
    """
    Return the ASCII diagram. Optionally "highlight" a state by wrapping it in brackets.
    """
    def H(name: str) -> str:
        if highlight == name:
            return f"[{name}]"
        return name

    q = [1.0 - p for p in ps]
    # format probabilities nicely
    fp = [f"{p:.3f}" for p in ps]
    fq = [f"{x:.3f}" for x in q]

    # Build a readable fixed diagram
    diagram = f"""6-Leg Parlay State Space (Directional Decay)

Legend:
  p_i  = P(win leg i)     q_i = 1 - p_i = P(lose leg i)
  S_k  = alive after k legs cleared (k wins in a row)
  BUST and JACKPOT are absorbing

     win p1={fp[0]}     win p2={fp[1]}     win p3={fp[2]}     win p4={fp[3]}     win p5={fp[4]}     win p6={fp[5]}
{H("S0")} ─────────▶ {H("S1")} ─────────▶ {H("S2")} ─────────▶ {H("S3")} ─────────▶ {H("S4")} ─────────▶ {H("S5")} ─────────▶ {H("JACKPOT")}
│             │             │             │             │             │
│ lose q1={fq[0]}  │ lose q2={fq[1]}  │ lose q3={fq[2]}  │ lose q4={fq[3]}  │ lose q5={fq[4]}  │ lose q6={fq[5]}
▼             ▼             ▼             ▼             ▼             ▼
{H("BUST")}  ◀──────┴─────────────┴─────────────┴─────────────┴─────────────┴───────────── (absorbing)

Constraints:
- Only forward transitions among S-states (no backward edges, no resets).
- Any loss transitions to BUST immediately.
- Winning leg 6 transitions to JACKPOT.
"""
    return diagram


def dot_graph(ps: List[float], highlight: Optional[str], show_bust_edges: bool) -> str:
    """
    Create a Graphviz DOT string for the parlay chain.
    """
    q = [1.0 - p for p in ps]
    # node style
    def node_style(n: str) -> str:
        base = 'shape=box'
        if n in ABSORBING:
            base = 'shape=doublecircle'
        if highlight == n:
            base += ', style="filled", fillcolor="lightyellow"'
        return base

    lines = ["digraph G {", 'rankdir="LR";', 'labelloc="t";',
             'label="6-Leg Parlay State Space (Directional Decay)";',
             'node [fontname="Helvetica"];', 'edge [fontname="Helvetica"];']

    # Nodes
    for n in STATE_NAMES:
        lines.append(f'{n} [{node_style(n)}];')

    # Forward win edges S0->S1...S4->S5
    for i in range(5):
        lines.append(f'{TRANSIENT[i]} -> {TRANSIENT[i+1]} [label="win p{i+1}={ps[i]:.3f}"];')

    # S5 -> JACKPOT
    lines.append(f'S5 -> JACKPOT [label="win p6={ps[5]:.3f}"];')

    # Bust edges
    if show_bust_edges:
        for i in range(6):
            lines.append(f'{TRANSIENT[i]} -> BUST [label="lose q{i+1}={q[i]:.3f}"];')

    # Absorbing self-loops (optional but clarifies absorbing nature)
    lines.append('BUST -> BUST [label="1", color="gray"];')
    lines.append('JACKPOT -> JACKPOT [label="1", color="gray"];')

    # Align absorbing nodes visually
    lines.append('{rank=same; BUST; JACKPOT;}')

    lines.append("}")
    return "\n".join(lines)


def extract_pdf_text(file) -> str:
    """
    Best-effort PDF text extraction using PyPDF2.
    If extraction fails, returns a helpful message.
    """
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
st.set_page_config(page_title="6-Parlay State Space Explorer", layout="wide")

st.title("6-Parlay State Space Explorer (Directional Decay)")
st.write(
    "Interactive analysis of the **state space** for a **6-leg parlay** modeled as a directional-decay Markov chain. "
    "Tune per-leg win probabilities, visualize transitions, and compute key metrics (Jackpot probability, expected life)."
)

with st.sidebar:
    st.header("Model Parameters")
    st.caption("Adjust per-leg win probabilities `p1..p6`.")

    default_p = 0.55
    ps = []
    for i in range(6):
        ps.append(
            st.slider(
                f"p{i+1} (win prob leg {i+1})",
                min_value=0.0,
                max_value=1.0,
                value=default_p,
                step=0.01
            )
        )

    st.divider()
    highlight = st.selectbox(
        "Highlight a state",
        options=["(none)"] + STATE_NAMES,
        index=0
    )
    highlight_state = None if highlight == "(none)" else highlight

    show_bust_edges = st.checkbox("Show BUST transitions", value=True)

    st.divider()
    st.header("Optional: Load Problem Set PDF")
    uploaded_pdf = st.file_uploader("Upload a PDF (optional)", type=["pdf"])
    st.caption("If text extraction works, the PDF text will be shown in the app.")

# Build core objects
P = build_transition_matrix(ps)
metrics = absorbing_chain_metrics(P)
diagram_txt = ascii_diagram(ps, highlight=highlight_state)
dot = dot_graph(ps, highlight=highlight_state, show_bust_edges=show_bust_edges)

tab1, tab2, tab3 = st.tabs(["State Space", "Metrics", "Problem Set (optional)"])

with tab1:
    colA, colB = st.columns([1, 1])

    with colA:
        st.subheader("ASCII Diagram")
        st.code(diagram_txt, language="text")

        with st.expander("State definitions (quick reference)", expanded=True):
            st.markdown(
                """
- **S0**: Start / 0 legs cleared (alive)  
- **S1**: 1 leg cleared (alive)  
- **S2**: 2 legs cleared (alive)  
- **S3**: 3 legs cleared (alive)  
- **S4**: 4 legs cleared (alive)  
- **S5**: 5 legs cleared (alive; final leg next)  
- **BUST**: terminal failure (absorbing)  
- **JACKPOT**: terminal success (absorbing)
                """.strip()
            )

    with colB:
        st.subheader("Graph View")
        st.graphviz_chart(dot, use_container_width=True)

        with st.expander("How directional decay works", expanded=True):
            st.markdown(
                """
**Directional**: transitions only move forward along the chain (`S0→S1→...→S5→JACKPOT`).  
**Decay**: from every live state `S_k` there’s a one-way “leak” to **BUST** with probability `q_{k+1}=1−p_{k+1}`.  
No backward edges, no resets—once you hit **BUST** or **JACKPOT**, you stay there.
                """.strip()
            )

with tab2:
    st.subheader("Key Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jackpot Probability", f"{metrics.jackpot_prob:.4f}")
    c2.metric("Bust Probability", f"{metrics.bust_prob:.4f}")
    c3.metric("Expected Legs Won", f"{metrics.expected_legs_survived:.3f}")
    c4.metric("Expected Steps to Absorption", f"{metrics.expected_steps_to_absorption:.3f}")

    st.divider()
    st.subheader("Transition Matrix (P)")
    st.caption("States ordered: S0,S1,S2,S3,S4,S5,BUST,JACKPOT")
    st.dataframe(
        P,
        use_container_width=True,
        column_config={i: STATE_NAMES[i] for i in range(len(STATE_NAMES))}
    )

    with st.expander("Interpretation notes", expanded=False):
        st.markdown(
            """
- **Jackpot Probability** is the probability of winning all 6 legs in order.
- **Expected Legs Won** is the expected number of successful legs before termination (either BUST or JACKPOT).
- **Expected Steps to Absorption** is the expected number of transitions until reaching an absorbing state.
            """.strip()
        )

with tab3:
    st.subheader("Problem Set PDF (optional)")
    if uploaded_pdf is None:
        st.info("Upload the PDF in the sidebar to view extracted text here.")
    else:
        st.caption("Best-effort text extraction (scanned PDFs may not extract well without OCR).")
        text = extract_pdf_text(uploaded_pdf)
        st.text_area("Extracted text", value=text, height=450)

st.caption("Run locally with: `streamlit run app.py`")

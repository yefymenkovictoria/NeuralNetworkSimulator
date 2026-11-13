"""Decode It with Vic: Spam Neuron Simulator.

A one-neuron "network" that decides whether an email is spam.
Built for the video "How Does AI Actually Learn? From Neural Networks to Hallucinations".
Neuron math follows IBM's neural network explainer (Fangfang Lee):
    z = (w1*x1 + w2*x2 + ... ) + b      (weighted sum plus bias)
    a = sigmoid(z) = 1 / (1 + e^-z)     (activation: squashes z into 0..1)
"""

import math

import pandas as pd
import streamlit as st

WORDS = ["prize", "money", "dear", "win", "hello"]
DEFAULTS = {"prize": 3.0, "money": 2.0, "dear": 1.0, "win": 2.5, "hello": 0.1}
DEFAULT_BIAS = -3.0
DEFAULT_THRESHOLD = 80
DEFAULT_PRESENT = {"prize": True, "money": False, "dear": True, "win": True, "hello": False}

EMAILS = {
    "Dear friend, you have WON a free PRIZE!": {"prize": True, "money": False, "dear": True, "win": True, "hello": False},
    "Hello! Are we still on for study group?": {"prize": False, "money": False, "dear": False, "win": False, "hello": True},
    "Dear winner, send money to claim your prize!": {"prize": True, "money": True, "dear": True, "win": True, "hello": False},
}


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def reset_all():
    for w in WORDS:
        st.session_state[f"w_{w}"] = DEFAULTS[w]
        st.session_state[f"x_{w}"] = DEFAULT_PRESENT[w]
    st.session_state["bias"] = DEFAULT_BIAS
    st.session_state["threshold"] = DEFAULT_THRESHOLD


def load_email(name: str):
    for w in WORDS:
        st.session_state[f"x_{w}"] = EMAILS[name][w]


st.set_page_config(page_title="Spam Neuron Simulator", page_icon="🧠", layout="centered")

# Initialise state once.
for w in WORDS:
    st.session_state.setdefault(f"w_{w}", DEFAULTS[w])
    st.session_state.setdefault(f"x_{w}", DEFAULT_PRESENT[w])
st.session_state.setdefault("bias", DEFAULT_BIAS)
st.session_state.setdefault("threshold", DEFAULT_THRESHOLD)

st.title("🧠 Spam Neuron Simulator")
st.caption("From *Decode It with Vic*: How Does AI Actually Learn? Turn the dials and watch one neuron decide.")

st.markdown(
    "A neural network looks at the **words** in an email, decides how much each word matters "
    "(its **weight**), and gives a **spam probability**. "
    "You are the one turning the dials now."
)

# ---------------------------------------------------------------- Step 1
st.header("Step 1: Pick an email")
cols = st.columns(len(EMAILS))
for col, name in zip(cols, EMAILS):
    col.button(name, on_click=load_email, args=(name,), width="stretch")

st.write("Or tick the words yourself. A word that is present counts as **1**, absent counts as **0**.")
pcols = st.columns(len(WORDS))
for col, w in zip(pcols, WORDS):
    col.checkbox(w, key=f"x_{w}")

# ---------------------------------------------------------------- Step 2
st.header("Step 2: Turn the dials (weights)")
st.info(
    "A weight is a dial for how strongly a word pushes toward **spam** (positive) "
    "or toward **not spam** (negative). Zero means the neuron ignores the word."
)
for w in WORDS:
    st.slider(f"Weight for “{w}”", -5.0, 5.0, step=0.1, key=f"w_{w}")

st.slider(
    "Bias (how easily the neuron gets convinced)",
    -8.0, 8.0, step=0.1, key="bias",
    help="Negative bias = hard to convince. Positive bias = jumpy, flags almost anything.",
)
st.slider("Flag the email if spam probability is at least (%)", 50, 99, key="threshold")

# ---------------------------------------------------------------- Compute
rows = []
z = 0.0
for w in WORDS:
    x = 1 if st.session_state[f"x_{w}"] else 0
    wt = st.session_state[f"w_{w}"]
    z += x * wt
    rows.append({"Word": w, "Present (x)": x, "Weight (w)": wt, "Contribution (x × w)": round(x * wt, 2)})
bias = st.session_state["bias"]
z += bias
prob = sigmoid(z)
pct = prob * 100
flagged = pct >= st.session_state["threshold"]

# ---------------------------------------------------------------- Step 3
st.header("Step 3: Watch the neuron work")

st.subheader("A. Multiply and add")
st.write("Each word present is multiplied by its weight. Then we add everything up and add the bias.")
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.latex(r"z = \sum w_i x_i + b")
st.markdown(f"**z = {z - bias:.2f} + ({bias:.2f}) = {z:.2f}**")

st.subheader("B. Squash it into a probability")
st.write(
    "z can be any number, like -7 or 12. The activation function (sigmoid) squashes it "
    "into a number between 0% and 100%."
)
st.latex(r"a = \sigma(z) = \frac{1}{1 + e^{-z}}")
st.markdown(f"**a = σ({z:.2f}) = {pct:.1f}% spam**")

st.subheader("C. Decide")
st.progress(min(max(prob, 0.0), 1.0), text=f"Spam probability: {pct:.1f}%  (flag line: {st.session_state['threshold']}%)")
if flagged:
    st.error(f"🚩 FLAGGED AS SPAM ({pct:.1f}%)")
else:
    st.success(f"✅ NOT FLAGGED ({pct:.1f}%)")

# ---------------------------------------------------------------- Curve
with st.expander("See where you are on the sigmoid curve"):
    xs = [i / 4 for i in range(-40, 41)]
    curve = pd.DataFrame({"z": xs, "spam probability (%)": [sigmoid(v) * 100 for v in xs]}).set_index("z")
    st.line_chart(curve)
    st.caption(f"Your z is {z:.2f}. Big negative z means near 0%. Big positive z means near 100%.")

# ---------------------------------------------------------------- Challenges
st.header("Try these challenges")
st.markdown(
    "1. Tick only **hello**. Can you make it spam by cranking its weight? Why is that a bad idea?\n"
    "2. Tick **prize**, **win**, **dear**. Lower the **prize** weight to 0. What happens to the probability?\n"
    "3. Make the bias very positive. What happens to a harmless email?\n"
    "4. Find weights where the first email is flagged but the study-group email is not."
)

with st.expander("What about learning? (Guess. Check. Adjust.)"):
    st.markdown(
        "You just did what training does. You **guessed** (set weights), **checked** (looked at the probability), "
        "and **adjusted** (moved a dial). A real network measures its mistake with a **loss function**, "
        "uses **backpropagation** to find which weight caused it, and nudges each weight with "
        "**gradient descent**: w_new = w − η × ∂L/∂w, where η (eta) is the learning rate. "
        "It does this millions of times, automatically."
    )

st.button("Reset everything", on_click=reset_all)
st.caption("Neuron math: Lee, Fangfang. “What Is a Neural Network?” IBM Think. Educational demo, not a real spam filter.")

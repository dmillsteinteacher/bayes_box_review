import streamlit as st
import numpy as np
import pandas as pd

# --- CONFIGURATION ---
STATES = ["State 1", "State 2", "State 3"]
OBSERVATIONS = ["Low", "Medium", "High"]

st.set_page_config(page_title="Bayes Box Review", layout="wide")

st.title("📦 The Bayes Box Review")
st.write("Understand how a single piece of evidence updates your prior belief.")

# --- 1. INPUTS: THE PRIOR ---
st.header("1. The Prior")
st.write("What do you believe *before* seeing the new data?")
c1, c2, c3 = st.columns(3)
p1 = c1.number_input("Prior State 1", 0.0, 1.0, 0.33, step=0.01)
p2 = c2.number_input("Prior State 2", 0.0, 1.0, 0.33, step=0.01)
p3 = c3.number_input("Prior State 3", 0.0, 1.0, 0.34, step=0.01)

prior = np.array([p1, p2, p3])
# Normalize prior if student enters weird numbers
if not np.isclose(prior.sum(), 1.0):
    st.warning("Prior does not sum to 1. Auto-normalizing...")
    prior = prior / prior.sum()

# --- 2. INPUTS: THE THEORY (B MATRIX) ---
st.header("2. The Theory (Emission Matrix)")
st.write("If the system is in a certain state, how likely is each observation?")
default_b = pd.DataFrame(
    [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], 
    columns=OBSERVATIONS, 
    index=STATES
)
b_matrix_df = st.data_editor(default_b, key="bayes_b")
B = b_matrix_df.to_numpy()

# --- 3. THE EVIDENCE ---
st.header("3. The Evidence")
selected_obs = st.selectbox("Select the Observation received:", OBSERVATIONS)
obs_idx = OBSERVATIONS.index(selected_obs)

# --- 4. THE CALCULATION (THE BOX) ---
st.divider()
st.header("4. The Update (Posterior)")

# Likelihoods for the specific observation
likelihoods = B[:, obs_idx]

# Unnormalized Posterior
unnorm_posterior = prior * likelihoods
total_evidence = np.sum(unnorm_posterior)

if total_evidence > 0:
    posterior = unnorm_posterior / total_evidence
else:
    posterior = np.array([0.0, 0.0, 0.0])

# --- 5. VISUALIZATION ---
res_df = pd.DataFrame({
    "Prior": prior,
    "Likelihood (P(Data|State))": likelihoods,
    "Unnormalized": unnorm_posterior,
    "Posterior": posterior
}, index=STATES)

st.table(res_df.style.format("{:.4f}"))

st.write("### Visual Shift")
chart_data = pd.DataFrame({
    "Prior": prior,
    "Posterior": posterior
}, index=STATES)
st.bar_chart(chart_data)

st.info(f"**Total Probability of observing '{selected_obs}' given your model:** {total_evidence:.4f}")
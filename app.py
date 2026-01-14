import streamlit as st
import numpy as np
import pandas as pd

# --- CONFIGURATION ---
STATES = ["State 1", "State 2", "State 3"]
OBSERVATIONS = ["Low", "Medium", "High"]

st.set_page_config(page_title="Iterative Bayes Box", layout="wide")

# --- SESSION STATE FOR ITERATION ---
if 'prior' not in st.session_state:
    st.session_state.prior = np.array([0.33, 0.33, 0.34])
if 'obs_history' not in st.session_state:
    st.session_state.obs_history = []

# --- SIDEBAR: RESET & THEORY ---
with st.sidebar:
    st.header("Theory Control")
    if st.button("Reset to Uniform Prior"):
        st.session_state.prior = np.array([0.33, 0.33, 0.34])
        st.session_state.obs_history = []
        st.rerun()
    
    st.write("### Emission Theory (B)")
    # Defaulting to a noisy sensor
    default_b = pd.DataFrame(
        [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]], 
        columns=OBSERVATIONS, 
        index=STATES
    )
    b_matrix_df = st.data_editor(default_b, key="iter_b")
    B = b_matrix_df.to_numpy()

# --- MAIN UI ---
st.title("🔄 Iterative Bayes Box")
st.write("Watch your belief sharpen as evidence accumulates over time (No Drift).")

# --- 1. CURRENT BELIEF & INPUT ---
c1, c2 = st.columns([1, 2])

with c1:
    st.subheader("Current Prior")
    # Show the current state of belief
    st.table(pd.DataFrame([st.session_state.prior], columns=STATES, index=["P(x)"]))
    
    st.subheader("Next Observation")
    selected_obs = st.selectbox("What did you hear?", OBSERVATIONS)
    
    if st.button("Update Belief", type="primary"):
        obs_idx = OBSERVATIONS.index(selected_obs)
        likelihoods = B[:, obs_idx]
        
        # The Bayes Calculation
        unnorm = st.session_state.prior * likelihoods
        total_ev = np.sum(unnorm)
        
        if total_ev > 0:
            posterior = unnorm / total_ev
            
            # Record for history
            st.session_state.obs_history.append({
                "Observation": selected_obs,
                "Likelihoods": [round(x, 3) for x in likelihoods],
                "Posterior": [round(x, 3) for x in posterior]
            })
            
            # Update Prior for next iteration
            st.session_state.prior = posterior
            st.rerun()
        else:
            st.error("Likelihood is zero for all states! Evidence is impossible under current theory.")

with c2:
    st.subheader("Belief Distribution")
    chart_df = pd.DataFrame(st.session_state.prior, index=STATES, columns=["Probability"])
    st.bar_chart(chart_df, color="#ff4b4b")

# --- 2. THE HISTORY TABLE ---
if st.session_state.obs_history:
    st.divider()
    st.subheader("📜 Evidence Log")
    history_df = pd.DataFrame(st.session_state.obs_history)
    st.table(history_df)

    

# --- 3. THE "LIGHTBULB" MOMENT ---
st.info("""
**Observation for Seniors:** Notice how the 'Posterior' from the previous row is used as the 'Prior' for the next. 
If you get three 'High' observations in a row, does your certainty reach 100%? 
How does the noise in your B Matrix (the off-diagonal numbers) prevent you from being overconfident?
""")

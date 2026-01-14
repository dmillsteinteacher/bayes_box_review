import streamlit as st
import numpy as np
import pandas as pd

# --- CONFIGURATION ---
STATES = ["State 1", "State 2", "State 3"]
OBSERVATIONS = ["Low", "Medium", "High"]

st.set_page_config(page_title="Iterative Bayes Box", layout="wide")

# --- SESSION STATE ---
if 'current_prior' not in st.session_state:
    st.session_state.current_prior = np.array([0.333, 0.333, 0.334])
if 'history' not in st.session_state:
    st.session_state.history = []

# --- SIDEBAR: THEORY & RESET ---
with st.sidebar:
    st.header("Theory Configuration")
    if st.button("Reset to Uniform Prior"):
        st.session_state.current_belief = np.array([0.333, 0.333, 0.334])
        st.session_state.history = []
        st.rerun()
    
    st.write("### Emission Matrix (B)")
    # Default theory: Distinct but noisy signals
    default_b = pd.DataFrame(
        [[0.8, 0.1, 0.1], 
         [0.1, 0.8, 0.1], 
         [0.1, 0.1, 0.8]], 
        columns=OBSERVATIONS, 
        index=STATES
    )
    b_df = st.data_editor(default_b, key="theory_b")
    B_matrix = b_df.to_numpy()

# --- MAIN INTERFACE ---
st.title("🔄 Iterative Bayes Box Analysis")

# Step 1: Input Evidence
st.header("1. Current Evidence")
selected_obs = st.selectbox("Select the new observation:", OBSERVATIONS)
obs_idx = OBSERVATIONS.index(selected_obs)

if st.button("Calculate Bayes Update", type="primary"):
    # Perform the Bayes Box Math
    likelihoods = B_matrix[:, obs_idx]
    unnormalized = st.session_state.current_prior * likelihoods
    total_evidence = np.sum(unnormalized)
    
    if total_evidence > 0:
        posterior = unnormalized / total_evidence
        
        # Save the specific box for this step before updating the prior
        step_box = pd.DataFrame({
            "Prior P(H)": st.session_state.current_prior,
            "Likelihood P(D|H)": likelihoods,
            "Unnormalized P(H)P(D|H)": unnormalized,
            "Posterior P(H|D)": posterior
        }, index=STATES)
        
        # Record history
        st.session_state.history.insert(0, {
            "obs": selected_obs,
            "box": step_box,
            "total_ev": total_evidence
        })
        
        # Update the prior for the next step
        st.session_state.current_prior = posterior
    else:
        st.error("The observation is impossible given your current theory (Total Evidence = 0).")

# Step 2: Display the Calculation and History
st.divider()

if not st.session_state.history:
    st.info("Select an observation and click 'Calculate' to begin the iteration.")
    st.write("### Starting Prior")
    st.table(pd.DataFrame([st.session_state.current_prior], columns=STATES, index=["P(H)"]))
else:
    # Show the most recent calculation as the primary focus
    latest = st.session_state.history[0]
    
    st.header(f"2. Calculation for Evidence: '{latest['obs']}'")
    st.write(f"**Total Probability of Evidence P(D):** {latest['total_ev']:.4f}")
    
    # Display the actual Bayes Box
    st.table(latest['box'].style.format("{:.4f}"))

    

    # Show the shift visually
    st.subheader("Visual Shift")
    plot_df = latest['box'][["Prior P(H)", "Posterior P(H|D)"]]
    st.bar_chart(plot_df)

    # Historical Log
    if len(st.session_state.history) > 1:
        with st.expander("View Previous Iterations"):
            for i, record in enumerate(st.session_state.history[1:]):
                st.write(f"**Step {len(st.session_state.history)-1-i}: Observation '{record['obs']}'**")
                st.table(record['box'].style.format("{:.3f}"))

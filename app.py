import streamlit as st
import numpy as np
import pandas as pd

# --- CONFIGURATION ---
STATES = ["State 1", "State 2", "State 3"]
OBSERVATIONS = ["Low", "Medium", "High"]

st.set_page_config(page_title="Iterative Bayes Box", layout="wide")

# --- SESSION STATE ---
if 'locked' not in st.session_state:
    st.session_state.locked = False
if 'current_prior' not in st.session_state:
    st.session_state.current_prior = np.array([0.333, 0.333, 0.334])
if 'history' not in st.session_state:
    st.session_state.history = []
if 'B_matrix' not in st.session_state:
    st.session_state.B_matrix = np.eye(3)

# --- UTILITIES ---
def validate_rows(df):
    return np.allclose(df.sum(axis=1), 1.0, atol=1e-3)

# --- MAIN UI ---
st.title("🔄 Iterative Bayes Box Analysis")

# --- STEP 1: MODEL CONFIGURATION (UNLOCKED ONLY) ---
if not st.session_state.locked:
    st.header("1. Configure Your Model")
    st.write("Set your initial belief and your emissions theory, then lock them to begin.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("### Initial Prior Vector")
        init_p_df = st.data_editor(pd.DataFrame([[0.333, 0.333, 0.334]], columns=STATES), key="init_p_editor")
    with col_b:
        st.write("### Emissions Theory (B)")
        init_b_df = st.data_editor(pd.DataFrame(
            [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], 
            columns=OBSERVATIONS, index=STATES), key="init_b_editor")

    if st.button("✅ Validate and Lock Model", type="primary"):
        if validate_rows(init_p_df) and validate_rows(init_b_df):
            st.session_state.current_prior = init_p_df.to_numpy().flatten()
            st.session_state.B_matrix = init_b_df.to_numpy()
            st.session_state.locked = True
            st.rerun()
        else:
            st.error("Matrix rows must sum to 1.0. Please adjust your values.")
else:
    # --- STEP 2: ITERATIVE UPDATES (LOCKED ONLY) ---
    with st.sidebar:
        st.success("Model Locked")
        if st.button("🔓 Unlock & Reset"):
            st.session_state.locked = False
            st.session_state.history = []
            st.session_state.current_prior = np.array([0.333, 0.333, 0.334])
            st.rerun()
        
        st.write("### Active Emissions Theory")
        st.table(pd.DataFrame(st.session_state.B_matrix, columns=OBSERVATIONS, index=STATES))

    st.header("2. Process Evidence")
    selected_obs = st.selectbox("Select the new observation:", OBSERVATIONS)
    obs_idx = OBSERVATIONS.index(selected_obs)

    if st.button("Calculate Bayes Update"):
        likelihoods = st.session_state.B_matrix[:, obs_idx]
        unnormalized = st.session_state.current_prior * likelihoods
        total_evidence = np.sum(unnormalized)
        
        if total_evidence > 0:
            posterior = unnormalized / total_evidence
            
            # The Bayes Box
            step_box = pd.DataFrame({
                "Prior P(H)": st.session_state.current_prior,
                "Likelihood P(D|H)": likelihoods,
                "Unnormalized": unnormalized,
                "Posterior P(H|D)": posterior
            }, index=STATES)
            
            st.session_state.history.insert(0, {
                "obs": selected_obs,
                "box": step_box,
                "total_ev": total_evidence
            })
            st.session_state.current_prior = posterior
        else:
            st.error("Impossible evidence under current theory.")

    # --- DISPLAY RESULTS ---
    if st.session_state.history:
        latest = st.session_state.history[0]
        st.divider()
        st.write(f"### Bayes Box for Evidence: '{latest['obs']}'")
        st.table(latest['box'].style.format("{:.4f}"))
        
        
        
        st.bar_chart(latest['box'][["Prior P(H)", "Posterior P(H|D)"]])

        with st.expander("Full Iteration History"):
            for i, record in enumerate(st.session_state.history):
                st.write(f"**Iteration {len(st.session_state.history)-i}: {record['obs']}**")
                st.dataframe(record['box'])

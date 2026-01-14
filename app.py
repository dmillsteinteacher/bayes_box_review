import streamlit as st
import numpy as np
import pandas as pd

# --- INITIAL SETUP ---
st.set_page_config(page_title="Custom Bayes Lab", layout="wide")
EPSILON = 1e-6

# --- SESSION STATE ---
if 'setup_stage' not in st.session_state:
    st.session_state.setup_stage = "naming"  # naming -> configuring -> locked
if 'states' not in st.session_state:
    st.session_state.states = ["State 1", "State 2", "State 3"]
if 'observations' not in st.session_state:
    st.session_state.observations = ["Low", "Medium", "High"]
if 'current_prior' not in st.session_state:
    st.session_state.current_prior = None
if 'B_matrix' not in st.session_state:
    st.session_state.B_matrix = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'trend_data' not in st.session_state:
    st.session_state.trend_data = []

# --- UTILITIES ---
def protect_and_normalize(arr):
    arr = np.array(arr)
    arr[arr <= 0] = EPSILON
    return arr / arr.sum()

def check_normalization(df, name):
    errors = []
    matrix = df.to_numpy()
    row_sums = matrix.sum(axis=1)
    for i, s in enumerate(row_sums):
        if not np.isclose(s, 1.0, atol=1e-3):
            label = df.index[i] if len(df.index) > 1 else "Vector"
            errors.append(f"**{name}** ({label}): Sums to {s:.4f} (expected 1.0)")
    return errors

# --- STAGE 1: NAMING ---
if st.session_state.setup_stage == "naming":
    st.title("🏷️ 1. Name Your Variables")
    st.write("Define the hidden states and the observations you will receive.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Hidden States")
        s1 = st.text_input("State 1 Label", "State 1")
        s2 = st.text_input("State 2 Label", "State 2")
        s3 = st.text_input("State 3 Label", "State 3")
    with c2:
        st.subheader("Observations")
        o1 = st.text_input("Observation 1 Label", "Low")
        o2 = st.text_input("Observation 2 Label", "Medium")
        o3 = st.text_input("Observation 3 Label", "High")
    
    if st.button("Proceed to Theory Configuration"):
        st.session_state.states = [s1, s2, s3]
        st.session_state.observations = [o1, o2, o3]
        st.session_state.setup_stage = "configuring"
        st.rerun()

# --- STAGE 2: CONFIGURATION ---
elif st.session_state.setup_stage == "configuring":
    st.title("⚙️ 2. Configure Theory")
    st.write(f"Define your model for: **{', '.join(st.session_state.states)}**")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("### Initial Prior")
        init_p_df = st.data_editor(pd.DataFrame([[0.333, 0.333, 0.334]], columns=st.session_state.states, index=["Prior"]))
    with col_b:
        st.write("### Emissions Matrix (B)")
        init_b_df = st.data_editor(pd.DataFrame(
            [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], 
            columns=st.session_state.observations, index=st.session_state.states))

    if st.button("✅ Validate and Lock Model"):
        errors = check_normalization(init_p_df, "Prior") + check_normalization(init_b_df, "Emissions")
        if not errors:
            st.session_state.current_prior = protect_and_normalize(init_p_df.to_numpy().flatten())
            st.session_state.B_matrix = np.array([protect_and_normalize(row) for row in init_b_df.to_numpy()])
            st.session_state.trend_data = [{"Step": 0, "Obs": "Initial", **dict(zip(st.session_state.states, st.session_state.current_prior))}]
            st.session_state.setup_stage = "locked"
            st.rerun()
        else:
            for err in errors: st.error(err)

# --- STAGE 3: EXECUTION ---
else:
    with st.sidebar:
        st.success("Session Active")
        if st.button("🔄 Reset Everything"):
            st.session_state.setup_stage = "naming"
            st.session_state.history = []
            st.session_state.trend_data = []
            st.rerun()
        st.write("### Fixed Theory")
        st.table(pd.DataFrame(st.session_state.B_matrix, columns=st.session_state.observations, index=st.session_state.states))

    st.title(f"🔄 Analysis: {st.session_state.states[0]} vs {st.session_state.states[1]} vs {st.session_state.states[2]}")
    
    selected_obs = st.selectbox("Record New Evidence:", st.session_state.observations)
    obs_idx = st.session_state.observations.index(selected_obs)

    if st.button("Calculate Bayes Update"):
        likelihoods = protect_and_normalize(st.session_state.B_matrix[:, obs_idx])
        unnorm = st.session_state.current_prior * likelihoods
        total_ev = np.sum(unnorm)
        posterior = protect_and_normalize(unnorm)
        
        step_num = len(st.session_state.history) + 1
        step_box = pd.DataFrame({
            "Prior P(H)": st.session_state.current_prior,
            "Likelihood P(D|H)": likelihoods,
            "Unnormalized": unnorm,
            "Posterior P(H|D)": posterior
        }, index=st.session_state.states)
        
        st.session_state.history.insert(0, {"step": step_num, "obs": selected_obs, "box": step_box, "total_ev": total_ev})
        st.session_state.current_prior = posterior
        st.session_state.trend_data.append({"Step": step_num, "Obs": selected_obs, **dict(zip(st.session_state.states, posterior))})

    if st.session_state.history:
        latest = st.session_state.history[0]
        st.divider()
        st.subheader(f"Step {latest['step']}: Observed '{latest['obs']}'")
        st.table(latest['box'].style.format("{:.4f}"))
        
        
        
        st.write("### Probability Trend")
        trend_df = pd.DataFrame(st.session_state.trend_data).set_index("Step")
        st.line_chart(trend_df[st.session_state.states])

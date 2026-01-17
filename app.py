import streamlit as st
import numpy as np
import pandas as pd

st.write("Sat Jan 17, 15:58")

# --- INITIAL SETUP ---
st.set_page_config(page_title="HMM Strategy Lab", layout="wide")
EPSILON = 1e-6
MIN_PROB = 0.001  # The "Humility Constant" (Cromwell's Rule)

# --- SESSION STATE ---
if 'setup_stage' not in st.session_state:
    st.session_state.setup_stage = "naming"
if 'states' not in st.session_state:
    st.session_state.states = ["State A", "State B", "State C"]
if 'observations' not in st.session_state:
    st.session_state.observations = ["Clue 1", "Clue 2", "Clue 3"]
if 'current_prior' not in st.session_state:
    st.session_state.current_prior = np.array([0.333, 0.333, 0.334])
if 'A_matrix' not in st.session_state:
    st.session_state.A_matrix = None
if 'B_matrix' not in st.session_state:
    st.session_state.B_matrix = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'trend_data' not in st.session_state:
    st.session_state.trend_data = []

# --- UTILITIES ---
def protect_and_normalize(matrix_or_arr):
    """Ensures no value is zero (Cromwell's Rule) and renormalizes."""
    arr = np.array(matrix_or_arr)
    was_zero = np.any(arr <= 0)
    arr = np.maximum(arr, MIN_PROB)
    
    if arr.ndim == 1:
        return arr / arr.sum(), was_zero
    else:
        row_sums = arr.sum(axis=1)
        return arr / row_sums[:, np.newaxis], was_zero

def calculate_certainty(probs):
    n = len(probs)
    if n <= 1: return 1.0
    h = -np.sum(probs * np.log2(probs + EPSILON))
    h_max = np.log2(n)
    certainty = 1.0 - (h / h_max)
    return np.clip(certainty, 0.0, 1.0)

def check_normalization(df, name):
    errors = []
    matrix = df.to_numpy()
    row_sums = matrix.sum(axis=1)
    for i, s in enumerate(row_sums):
        if not np.isclose(s, 1.0, atol=1e-2):
            label = df.index[i]
            errors.append(f"**{name}** ({label}) must sum to 1.0. (Current: {s:.3f})")
    return errors

def convert_history_to_csv():
    if not st.session_state.history: return None
    frames = []
    for record in reversed(st.session_state.history):
        df = record['box'].copy()
        df.insert(0, 'Action Number', record['step'])
        df.insert(1, 'Action Type', record['action'])
        df.index.name = 'State'
        frames.append(df)
    return pd.concat(frames).to_csv().encode('utf-8')

# --- STAGE 1: NAMING ---
if st.session_state.setup_stage == "naming":
    st.title("🏷️ 1. Set the Scene")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🕵️ The Suspects (Hidden States)")
        s1 = st.text_input("State 1", st.session_state.states[0])
        s2 = st.text_input("State 2", st.session_state.states[1])
        s3 = st.text_input("State 3", st.session_state.states[2])
    with c2:
        st.subheader("🔎 The Clues (Observations)")
        o1 = st.text_input("Clue 1", st.session_state.observations[0])
        o2 = st.text_input("Clue 2", st.session_state.observations[1])
        o3 = st.text_input("Clue 3", st.session_state.observations[2])
    
    if st.button("Configure Movement & Theory ➡️"):
        st.session_state.states = [s1, s2, s3]
        st.session_state.observations = [o1, o2, o3]
        st.session_state.setup_stage = "configuring"
        st.rerun()

# --- STAGE 2: CONFIGURATION ---
elif st.session_state.setup_stage == "configuring":
    st.title("⚙️ 2. Configure Your World")
    
    st.subheader("🔄 Movement Rules (Transition Matrix)")
    stability = st.select_slider(
        "Set System Stability (Presets)",
        options=["Very Chaotic", "Fluid", "Stable", "Highly Inert"],
        value="Stable"
    )
    
    diag_map = {"Very Chaotic": 0.33, "Fluid": 0.6, "Stable": 0.85, "Highly Inert": 0.98}
    d = diag_map[stability]
    off_d = (1.0 - d) / 2
    default_a = [[d, off_d, off_d], [off_d, d, off_d], [off_d, off_d, d]]
    
    st.caption("Manual Tweak: Each row defines: 'If in State X, where do I go next?'")
    init_a_df = st.data_editor(pd.DataFrame(default_a, columns=st.session_state.states, index=st.session_state.states))

    st.divider()

    col_prior, col_theory = st.columns([1, 2])
    with col_prior:
        st.subheader("🏠 Initial Prior")
        init_p_df = st.data_editor(pd.DataFrame([[0.33, 0.33, 0.34]], columns=st.session_state.states, index=["Prob"]))
        
    with col_theory:
        st.subheader("📖 Clue Dictionary (Emissions)")
        init_b_df = st.data_editor(pd.DataFrame(
            [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7]], 
            columns=st.session_state.observations, index=st.session_state.states))

    if st.button("✅ Lock Model & Enter Dashboard"):
        errors = check_normalization(init_a_df, "Transitions") + \
                 check_normalization(init_p_df, "Prior") + \
                 check_normalization(init_b_df, "Theory")
        if not errors:
            st.session_state.A_matrix, a_zero = protect_and_normalize(init_a_df.to_numpy())
            st.session_state.B_matrix, b_zero = protect_and_normalize(init_b_df.to_numpy())
            st.session_state.current_prior, p_zero = protect_and_normalize(init_p_df.to_numpy().flatten())
            
            st.session_state.trend_data = [{"Action Number": 0, "Type": "Initial", **dict(zip(st.session_state.states, st.session_state.current_prior))}]
            st.session_state.setup_stage = "locked"
            
            if a_zero or b_zero or p_zero:
                st.toast(f"Note: Probabilities of 0% adjusted to {MIN_PROB:.1%} for model flexibility.", icon="⚠️")
            
            st.rerun()
        else:
            for err in errors: st.error(err)

# --- STAGE 3: DASHBOARD ---
else:
    with st.sidebar:
        st.header("⚙️ World Settings")
        st.write("**Transitions (A)**")
        st.dataframe(pd.DataFrame(st.session_state.A_matrix, columns=st.session_state.states, index=st.session_state.states).style.format("{:.3f}"))
        st.write("**Emissions (B)**")
        st.dataframe(pd.DataFrame(st.session_state.B_matrix, columns=st.session_state.observations, index=st.session_state.states).style.format("{:.3f}"))
        if st.button("🔄 Reset System"):
            st.session_state.setup_stage = "naming"
            st.session_state.history = []
            st.session_state.trend_data = []
            st.session_state.current_prior = np.array([0.333, 0.333, 0.334])
            st.rerun()

    st.title("🔮 Manual HMM Forecaster")
    
    col_ctrl, col_viz = st.columns([1, 2])
    
    with col_ctrl:
        st.subheader("🎮 Manual Controls")
        with st.container(border=True):
            st.write("#### Phase 1: Pass Time")
            if st.button("⏳ Advance 1 Time Step (Markov)", use_container_width=True):
                new_state = st.session_state.current_prior @ st.session_state.A_matrix
                step_box = pd.DataFrame({"Before": st.session_state.current_prior, "After": new_state}, index=st.session_state.states)
                step_num = len(st.session_state.history) + 1
                st.session_state.history.insert(0, {"step": step_num, "action": "Time Step", "box": step_box, "obs": "None"})
                st.session_state.current_prior = new_state
                st.session_state.trend_data.append({"Action Number": step_num, "Type": "Time", **dict(zip(st.session_state.states, new_state))})
                st.rerun()

        with st.container(border=True):
            st.write("#### Phase 2: Record Clue")
            selected_obs = st.selectbox("Observed Clue:", st.session_state.observations)
            if st.button("🔎 Update with Evidence (Bayes)", use_container_width=True):
                obs_idx = st.session_state.observations.index(selected_obs)
                likelihoods = st.session_state.B_matrix[:, obs_idx]
                unnorm = st.session_state.current_prior * likelihoods
                total_ev = np.sum(unnorm)
                posterior = unnorm / total_ev if total_ev > 0 else unnorm
                
                bayes_box = pd.DataFrame({
                    "Prior P(H)":

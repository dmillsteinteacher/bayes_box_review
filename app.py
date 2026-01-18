import streamlit as st
import numpy as np
import pandas as pd

# --- INITIAL SETUP ---
st.set_page_config(page_title="Bayes Box Forecaster", layout="wide")
EPSILON = 1e-6
MIN_PROB = 0.001 

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
if 'initial_vector' not in st.session_state:
    st.session_state.initial_vector = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'trend_data' not in st.session_state:
    st.session_state.trend_data = []

# --- UTILITIES ---
def protect_and_normalize(matrix_or_arr):
    arr = np.array(matrix_or_arr)
    arr = np.maximum(arr, MIN_PROB)
    if arr.ndim == 1:
        return (arr / arr.sum())
    else:
        row_sums = arr.sum(axis=1)
        return (arr / row_sums[:, np.newaxis])

def calculate_certainty(probs):
    n = len(probs)
    if n <= 1: return 1.0
    h = -np.sum(probs * np.log2(probs + EPSILON))
    h_max = np.log2(n)
    return np.clip(1.0 - (h / h_max), 0.0, 1.0)

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
    
    if st.button("Configure Theory & Movement ➡️"):
        st.session_state.states = [s1, s2, s3]
        st.session_state.observations = [o1, o2, o3]
        st.session_state.setup_stage = "configuring"
        st.rerun()

# --- STAGE 2: CONFIGURATION ---
elif st.session_state.setup_stage == "configuring":
    st.title("⚙️ 2. Configure Your Theory")
    
    st.subheader("🏠 Initial Prior")
    init_p_df = st.data_editor(pd.DataFrame(
        [st.session_state.current_prior], 
        columns=st.session_state.states, 
        index=["Prior Probability"]
    ))

    st.divider()
    
    st.subheader("📖 Clue Dictionary (Likelihoods)")
    init_b_df = st.data_editor(pd.DataFrame(
        [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7]], 
        columns=st.session_state.observations, 
        index=st.session_state.states
    ))

    st.divider()

    st.subheader("🔄 Movement Rules (System Drift)")
    stability = st.select_slider(
        "System Stability", 
        options=["Chaotic", "Fluid", "Stable", "Inert (Identity)"], 
        value="Inert (Identity)"
    )
    d_m = {"Chaotic": 0.4, "Fluid": 0.7, "Stable": 0.9, "Inert (Identity)": 1.0}
    d = d_m[stability]
    off = (1.0 - d) / 2
    default_a = [[d, off, off], [off, d, off], [off, off, d]]
    init_a_df = st.data_editor(pd.DataFrame(
        default_a, 
        columns=st.session_state.states, 
        index=st.session_state.states
    ))

    if st.button("✅ Start Investigation"):
        st.session_state.A_matrix = protect_and_normalize(init_a_df.to_numpy())
        st.session_state.B_matrix = protect_and_normalize(init_b_df.to_numpy())
        st.session_state.current_prior = protect_and_normalize(init_p_df.to_numpy().flatten())
        st.session_state.initial_vector = st.session_state.current_prior.copy()
        
        start_pt = {"Action Number": 0, "Type": "Initial"}
        for i, name in enumerate(st.session_state.states): 
            start_pt[name] = st.session_state.current_prior[i]
        st.session_state.trend_data = [start_pt]
        st.session_state.setup_stage = "locked"
        st.rerun()

# --- STAGE 3: DASHBOARD ---
else:
    with st.sidebar:
        st.header("🕵️ Case Parameters")
        st.write("**Starting Prior**")
        st.dataframe(pd.DataFrame({"Prob": st.session_state.initial_vector}, index=st.session_state.states).style.format("{:.3f}"))
        st.write("**Movement Rules**")
        st.dataframe(pd.DataFrame(st.session_state.A_matrix, columns=st.session_state.states, index=st.session_state.states).style.format("{:.2f}"))
        st.write("**Likelihood Dictionary**")
        st.dataframe(pd.DataFrame(st.session_state.B_matrix, columns=st.session_state.observations, index=st.session_state.states).style.format("{:.2f}"))
        if st.button("🔄 New Scenario"):
            st.session_state.setup_stage = "naming"
            st.session_state.history, st.session_state.trend_data = [], []
            st.rerun()

    st.title("🔮 Bayes Box Based Forecaster")
    
    col_ctrl, col_viz = st.columns([1, 2])
    with col_ctrl:
        st.subheader("🎮 Record Evidence")
        
        with st.container(border=True):
            st.write("#### 🔎 Observe Clue")
            sel_obs = st.selectbox("Recorded Clue:", st.session_state.observations)
            if st.button("Apply Bayes Update", use_container_width=True):
                idx = st.session_state.observations.index(sel_obs)
                lk = st.session_state.B_matrix[:, idx]
                unnorm = st.session_state.current_prior * lk
                total = np.sum(unnorm)
                post = (unnorm / total) if total > 0 else unnorm
                sn = len(st.session_state.history) + 1
                
                res_box = pd.DataFrame({
                    "Prior P(H)": st.session_state.current_prior, 
                    "Likelihood P(D|H)": lk, 
                    "Unnormalized": unnorm,
                    "Posterior": post
                }, index=st.session_state.states)
                
                st.session_state.history.insert(0, {"step": sn, "action": "Evidence", "box": res_box, "obs": sel_obs})
                st.session_state.current_prior = post
                
                tr = {"Action Number": sn, "Type": "Clue"}
                for i, name in enumerate(st.session_state.states): tr[name] = post[i]
                st.session_state.trend_data.append(tr)
                st.rerun()

        with st.container(border=True):
            st.write("#### ⏳ Pass Time")
            if st.button("Advance 1 Time Step", use_container_width=True):
                new_state = st.session_state.current_prior @ st.session_state.A_matrix
                sn = len(st.session_state.history) + 1
                res_box = pd.DataFrame({
                    "Before Drift": st.session_state.current_prior, 
                    "After Drift": new_state
                }, index=st.session_state.states)
                st.session_state.history.insert(0, {"step": sn, "action": "Time", "box": res_box, "obs": "None"})
                st.session_state.current_prior = new_state
                tr = {"Action Number": sn, "Type": "Time"}
                for i, name in enumerate(st.session_state.states): tr[name] = new_state[i]
                st.session_state.trend_data.append(tr)
                st.rerun()

    with col_viz:
        st.subheader("📊 Current Forecast")
        st.bar_chart(pd.DataFrame({"Probability": st.session_state.current_prior}, index=st.session_state.states), height=250)
        st.metric("Certainty Gauge", f"{calculate_certainty(st.session_state.current_prior):.1%}")
        
    st.divider()
    t1, t2 = st.tabs(["📈 Probability Trend", "📂 Investigation Log"])
    with t1:
        if st.session_state.trend_data:
            st.line_chart(pd.DataFrame(st.session_state.trend_data).set_index("Action Number")[st.session_state.states])
    with t2:
        for r in st.session_state.history:
            st.write(f"**Action {r['step']}: {r['action']}** {f'({r['obs']})' if r['obs'] != 'None' else ''}")
            st.table(r['box'].style.format("{:.4f}"))

# --- END OF FILE BUFFER ---
# These lines ensure that a partial copy-paste does not terminate the functional code above.
# End of Investigation Log.
# File closing procedures complete.

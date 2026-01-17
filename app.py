import streamlit as st
import numpy as np
import pandas as pd

# --- INITIAL SETUP ---
st.set_page_config(page_title="Forecaster Strategy Lab", layout="wide")
EPSILON = 1e-6

# --- SESSION STATE ---
if 'setup_stage' not in st.session_state:
    st.session_state.setup_stage = "naming"
if 'states' not in st.session_state:
    st.session_state.states = ["State A", "State B", "State C"]
if 'observations' not in st.session_state:
    st.session_state.observations = ["Clue 1", "Clue 2", "Clue 3"]
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
            label = df.index[i]
            errors.append(f"**{name}** ({label}) must sum to 1.0. (Current: {s:.3f})")
    return errors

# --- STAGE 1: NAMING ---
if st.session_state.setup_stage == "naming":
    st.title("🏷️ 1. Set the Scene")
    st.info("Start by defining your 'Suspects' (States) and the 'Clues' (Observations) you will look for.")
    
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
    
    if st.button("Configure Theory ➡️"):
        st.session_state.states = [s1, s2, s3]
        st.session_state.observations = [o1, o2, o3]
        st.session_state.setup_stage = "configuring"
        st.rerun()

# --- STAGE 2: CONFIGURATION ---
elif st.session_state.setup_stage == "configuring":
    st.title("⚙️ 2. Build Your Theory")
    st.markdown("How do these clues relate to the suspects? Set your starting 'gut feeling' and your rules for evidence.")
    
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.write("### 🏠 Starting Gut Feeling (Prior)")
        init_p_df = st.data_editor(pd.DataFrame([[0.33, 0.33, 0.34]], columns=st.session_state.states, index=["Starting Probability"]))
        
    with col_b:
        st.write("### 📖 The Clue Dictionary (Emissions)")
        st.caption("If Suspect X is the truth, how often would they leave Clue Y? (Rows must sum to 1.0)")
        init_b_df = st.data_editor(pd.DataFrame(
            [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7]], 
            columns=st.session_state.observations, index=st.session_state.states))

    if st.button("✅ Lock Theory & Start Tracking"):
        errors = check_normalization(init_p_df, "Prior") + check_normalization(init_b_df, "Theory")
        if not errors:
            st.session_state.current_prior = protect_and_normalize(init_p_df.to_numpy().flatten())
            st.session_state.B_matrix = np.array([protect_and_normalize(row) for row in init_b_df.to_numpy()])
            st.session_state.trend_data = [{"Step": 0, "Clue": "Start", **dict(zip(st.session_state.states, st.session_state.current_prior))}]
            st.session_state.setup_stage = "locked"
            st.rerun()
        else:
            for err in errors: st.error(err)

# --- STAGE 3: DASHBOARD EXECUTION ---
else:
    with st.sidebar:
        st.header("🛠️ Model Control")
        st.write("### Your Theory Dictionary")
        st.table(pd.DataFrame(st.session_state.B_matrix, columns=st.session_state.observations, index=st.session_state.states))
        
        if st.button("🔄 Reset & Start Over"):
            st.session_state.setup_stage = "naming"
            st.session_state.history = []
            st.session_state.trend_data = []
            st.rerun()

    st.title("🔮 The Forecaster Dashboard")
    
    # TOP ROW: Controls and Current State
    top_left, top_right = st.columns([1, 2])
    
    with top_left:
        st.subheader("📥 Add New Evidence")
        selected_obs = st.selectbox("Choose a Clue to record:", st.session_state.observations)
        if st.button("Analyze This Clue ➡️", use_container_width=True):
            obs_idx = st.session_state.observations.index(selected_obs)
            likelihoods = st.session_state.B_matrix[:, obs_idx]
            
            # Bayes Update
            unnorm = st.session_state.current_prior * likelihoods
            total_ev = np.sum(unnorm)
            posterior = unnorm / total_ev if total_ev > 0 else unnorm
            
            # Save History
            step_num = len(st.session_state.history) + 1
            step_box = pd.DataFrame({
                "Before": st.session_state.current_prior,
                "Likelihood (Clue Strength)": likelihoods,
                "After": posterior
            }, index=st.session_state.states)
            
            st.session_state.history.insert(0, {"step": step_num, "obs": selected_obs, "box": step_box})
            st.session_state.current_prior = posterior
            st.session_state.trend_data.append({"Step": step_num, "Clue": selected_obs, **dict(zip(st.session_state.states, posterior))})
            st.rerun()

    with top_right:
        st.subheader("📊 Current Probability")
        # Display a bar chart of the current state
        current_df = pd.DataFrame({
            "Suspect": st.session_state.states,
            "Probability": st.session_state.current_prior
        })
        st.bar_chart(current_df.set_index("Suspect"), height=250)

    st.divider()

    # BOTTOM ROW: Trends and Logic
    bot_left, bot_right = st.columns([2, 1])
    
    with bot_left:
        st.subheader("📈 Probability Over Time")
        trend_df = pd.DataFrame(st.session_state.trend_data).set_index("Step")
        st.line_chart(trend_df[st.session_state.states])

    with bot_right:
        st.subheader("💡 Why did it change?")
        if st.session_state.history:
            latest = st.session_state.history[0]
            st.write(f"After seeing **{latest['obs']}**:")
            st.dataframe(latest['box'].style.highlight_max(axis=0, color='lightgreen').format("{:.2%}"))
            st.caption("Green highlights show the strongest evidence and the new leader.")
        else:
            st.write("No clues recorded yet. Use the box above to start your investigation.")

    # HISTORY EXPANDER
    with st.expander("📂 Complete Investigation Log"):
        for record in reversed(st.session_state.history): # Show oldest to newest here
            st.write(f"**Step {record['step']}: {record['obs']}**")
            st.dataframe(record['box'].style.format("{:.3f}"))

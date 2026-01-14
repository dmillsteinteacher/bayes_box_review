import streamlit as st
import numpy as np
import pandas as pd

# --- CONFIGURATION ---
STATES = ["State 1", "State 2", "State 3"]
OBSERVATIONS = ["Low", "Medium", "High"]
EPSILON = 1e-6  # The "Zero-Protection" Pad

st.set_page_config(page_title="Iterative Bayes Box", layout="wide")

# --- UTILITIES ---
def protect_and_normalize(arr):
    """Applies a small pad to zeros and re-normalizes to sum to 1.0."""
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

# --- SESSION STATE ---
if 'locked' not in st.session_state:
    st.session_state.locked = False
if 'current_prior' not in st.session_state:
    st.session_state.current_prior = np.array([0.333, 0.333, 0.334])
if 'history' not in st.session_state:
    st.session_state.history = []
if 'B_matrix' not in st.session_state:
    st.session_state.B_matrix = np.eye(3)
if 'trend_data' not in st.session_state:
    st.session_state.trend_data = []

# --- MAIN UI ---
st.title("🔄 Iterative Bayes Box Analysis")
st.info(f"**Zero-Protection Active:** Probabilities are padded by {EPSILON} to prevent logic traps.")

if not st.session_state.locked:
    st.header("1. Configure Initial Model")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("### Initial Prior Vector")
        init_p_df = st.data_editor(pd.DataFrame([[0.333, 0.333, 0.334]], columns=STATES, index=["Prior"]), key="init_p_edit")
    with col_b:
        st.write("### Emissions Theory (B)")
        init_b_df = st.data_editor(pd.DataFrame(
            [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], 
            columns=OBSERVATIONS, index=STATES), key="init_b_edit")

    if st.button("✅ Validate and Lock Model", type="primary"):
        errors = check_normalization(init_p_df, "Prior Vector") + check_normalization(init_b_df, "Emissions Matrix")
        if not errors:
            # Apply protection to the initial model setup
            st.session_state.current_prior = protect_and_normalize(init_p_df.to_numpy().flatten())
            st.session_state.B_matrix = np.array([protect_and_normalize(row) for row in init_b_df.to_numpy()])
            
            st.session_state.trend_data = [{"Step": 0, "Observation": "Initial", **dict(zip(STATES, st.session_state.current_prior))}]
            st.session_state.locked = True
            st.rerun()
        else:
            for err in errors: st.error(err)
else:
    with st.sidebar:
        st.success("Model Locked")
        if st.button("🔓 Unlock & Reset"):
            st.session_state.locked = False
            st.session_state.history = []
            st.session_state.trend_data = []
            st.rerun()
        st.write("### Protected Emissions Theory")
        st.table(pd.DataFrame(st.session_state.B_matrix, columns=OBSERVATIONS, index=STATES))

    st.header("2. Process Evidence")
    selected_obs = st.selectbox("Select current observation:", OBSERVATIONS)
    obs_idx = OBSERVATIONS.index(selected_obs)

    if st.button("Calculate Bayes Update"):
        # 1. Pull Likelihoods and apply protection
        likelihoods = protect_and_normalize(st.session_state.B_matrix[:, obs_idx])
        
        # 2. Compute Bayes Box
        unnormalized = st.session_state.current_prior * likelihoods
        
        # 3. Normalize the Posterior (Total Evidence check is safer now due to EPSILON)
        posterior = protect_and_normalize(unnormalized)
        total_ev = np.sum(unnormalized)
        
        step_num = len(st.session_state.history) + 1
        
        step_box = pd.DataFrame({
            "Prior P(H)": st.session_state.current_prior,
            "Likelihood P(D|H)": likelihoods,
            "Unnormalized": unnormalized,
            "Posterior P(H|D)": posterior
        }, index=STATES)
        
        st.session_state.history.insert(0, {"step": step_num, "observation": selected_obs, "box": step_box, "total_ev": total_ev})
        st.session_state.current_prior = posterior
        st.session_state.trend_data.append({"Step": step_num, "Observation": selected_obs, **dict(zip(STATES, posterior))})

    if st.session_state.history:
        latest = st.session_state.history[0]
        st.divider()
        st.subheader(f"Step {latest['step']} Result: Observed '{latest['observation']}'")
        st.write(f"**Total Probability of Evidence P(D):** {latest['total_ev']:.6f}")
        st.table(latest['box'].style.format("{:.6f}"))
        
        

        st.write("### Probability Trend Over Time")
        trend_df = pd.DataFrame(st.session_state.trend_data).set_index("Step")
        st.line_chart(trend_df[STATES])

        with st.expander("Full Iteration History"):
            for record in st.session_state.history:
                st.write(f"### Step {record['step']}: {record['observation']}")
                st.dataframe(record['box'].style.format("{:.6f}"))

import streamlit as st
import numpy as np
import pandas as pd

st.write("Sat Jan 17th, 15:55")

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
    """
    Ensures no value is zero (Cromwell's Rule) and renormalizes.
    """
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
        o1 = st.text_input("Clue 1", st

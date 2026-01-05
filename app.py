import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Traffic Light Optimization (ES)",
    layout="wide"
)

st.title("🚦 Traffic Light Optimization using Evolutionary Strategy (ES)")
st.markdown("Optimizing traffic signal timings to reduce congestion")

# ---------------- Load Dataset ----------------
df = pd.read_csv("traffic_dataset.csv")
baseline_waiting = df["waiting_time"].mean()

# ---------------- Sidebar Controls ----------------
st.sidebar.header("⚙️ ES Algorithm Parameters")

population_size = st.sidebar.slider(
    "Population Size (μ)",
    min_value=10,
    max_value=100,
    value=30,
    step=10
)

mutation_strength = st.sidebar.slider(
    "Mutation Strength (σ)",
    min_value=0.1,
    max_value=2.0,
    value=0.8,
    step=0.1
)

generations = st.sidebar.slider(
    "Number of Generations",
    min_value=10,
    max_value=100,
    value=40,
    step=10
)

# ---------------- Simulated ES Behaviour ----------------
np.random.seed(42)

fitness_history = []
current_fitness = baseline_waiting * 2

for g in range(generations):
    improvement = np.random.normal(loc=mutation_strength, scale=0.3)
    current_fitness = max(current_fitness - improvement, baseline_waiting * 0.4)
    fitness_history.append(cur_

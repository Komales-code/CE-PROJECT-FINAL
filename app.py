import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Traffic Signal Optimization using ES",
    layout="wide"
)

st.title("🚦 Traffic Signal Optimization using Evolution Strategy (ES)")
st.write("This application optimizes traffic signal green time to reduce congestion using Evolution Strategy.")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("traffic_dataset (1).csv")

data = load_data()

st.subheader("📊 Traffic Dataset Preview")
st.dataframe(data.head())

# -------------------------------
# Baseline Metrics
# -------------------------------
baseline_waiting_time = data["waiting_time"].mean()

st.subheader("⏱ Baseline Traffic Performance")
st.metric("Average Waiting Time (seconds)", f"{baseline_waiting_time:.2f}")

# -------------------------------
# Evolution Strategy Parameters
# -------------------------------
st.sidebar.header("⚙️ Evolution Strategy Parameters")

generations = st.sidebar.slider("Number of Generations", 20, 200, 100)
sigma = st.sidebar.slider("Mutation Step Size (σ)", 0.1, 10.0, 1.0)
min_green = st.sidebar.slider("Minimum Green Time (s)", 10, 30, 15)
max_green = st.sidebar.slider("Maximum Green Time (s)", 40, 120, 60)

# -------------------------------
# Fitness Function
# -------------------------------
def fitness_function(green_time, baseline_wait):
    """
    Simulated fitness function:
    - Assumes better green time allocation reduces waiting time
    """
    reduction_factor = np.exp(-green_time / max_green)
    optimized_wait = baseline_wait * reduction_factor
    return optimized_wait

# -------------------------------
# Evolution Strategy (1+1)
# -------------------------------
def evolution_strategy():
    parent = np.random.uniform(min_green, max_green)
    parent_fitness = fitness_function(parent, baseline_waiting_time)

    best_fitness_history = []
    best_solution_history = []

    for _ in range(generations):
        offspring = parent + np.random.normal(0, sigma)
        offspring = np.clip(offspring, min_green, max_green)

        offspring_fitness = fitness_function(offspring, baseline_waiting_time)

        if offspring_fitness < parent_fitness:
            parent, parent_fitness = offspring, offspring_fitness

        best_fitness_history.append(parent_fitness)
        best_solution_history.append(parent)

    return parent, parent_fitness, best_fitness_history, best_solution_history

# -------------------------------
# Run Optimization
# -------------------------------
if st.button("▶ Run Evolution Strategy Optimization"):
    best_green, best_fitness, fitness_curve, solution_curve = evolution_strategy()

    improvement = ((baseline_waiting_time - best_fitness) / baseline_waiting_time) * 100

    st.subheader("✅ Optimization Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Optimized Green Time (s)", f"{best_green:.2f}")
    col2.metric("Optimized Waiting Time (s)", f"{best_fitness:.2f}")
    col3.metric("Improvement (%)", f"{improvement:.2f}%")

    # -------------------------------
    # Results Table
    # -------------------------------
    result_df = pd.DataFrame({
        "Metric": ["Average Waiting Time (s)"],
        "Before ES": [baseline_waiting_time],
        "After ES": [best_fitness]
    })

    st.subheader("📋 Performance Comparison Table")
    st.table(result_df)

    # -------------------------------
    # Convergence Curve
    # -------------------------------
    st.subheader("📉 ES Convergence Curve")

    fig, ax = plt.subplots()
    ax.plot(fitness_curve)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (Waiting Time)")
    ax.set_title("Evolution Strategy Convergence")

    st.pyplot(fig)

    # -------------------------------
    # Solution Evolution
    # -------------------------------
    st.subheader("🧬 Evolution of Green Time")

    fig2, ax2 = plt.subplots()
    ax2.plot(solution_curve)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Green Time (s)")
    ax2.set_title("Green Time Adjustment over Generations")

    st.pyplot(fig2)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("**Course:** JIE42903 – Evolutionary Computing  ")
st.markdown("**Algorithm:** Evolution Strategy (ES)")

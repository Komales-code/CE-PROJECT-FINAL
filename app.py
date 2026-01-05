import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Page Config ----------------
st.set_page_config(page_title="Traffic Light Optimization (ES)", layout="wide")

st.title("🚦 Traffic Light Optimization using Evolutionary Strategy (ES)")
st.markdown("Dynamic optimization of traffic signal timings using real ES computation")

# ---------------- Load Dataset ----------------
df = pd.read_csv("traffic_dataset.csv")
baseline_waiting = df["waiting_time"].mean()

# ---------------- Sidebar Controls ----------------
st.sidebar.header("⚙️ ES Parameters")

population_size = st.sidebar.slider("Population Size (μ)", 10, 100, 30, 10)
mutation_strength = st.sidebar.slider("Mutation Strength (σ)", 0.01, 1.0, 0.2, 0.01)
generations = st.sidebar.slider("Generations", 10, 100, 40, 10)

# ---------------- Fitness Function ----------------
def fitness_function(multiplier):
    """
    Simulated waiting time based on signal timing multiplier
    """
    adjusted_waiting = baseline_waiting / multiplier
    noise = np.random.normal(0, 1)
    return max(adjusted_waiting + noise, 0)

# ---------------- Evolutionary Strategy ----------------
def evolutionary_strategy():
    population = np.random.uniform(0.5, 2.0, population_size)
    fitness_history = []

    for _ in range(generations):
        fitness_values = np.array([fitness_function(x) for x in population])
        best_index = np.argmin(fitness_values)
        best_solution = population[best_index]
        best_fitness = fitness_values[best_index]

        fitness_history.append(best_fitness)

        # Mutation
        population = best_solution + mutation_strength * np.random.randn(population_size)
        population = np.clip(population, 0.5, 2.0)

    return best_solution, best_fitness, fitness_history

# ---------------- Run Optimization ----------------
best_multiplier, optimized_waiting, fitness_history = evolutionary_strategy()

# ---------------- KPI Section ----------------
st.subheader("📊 Performance Summary")

col1, col2, col3 = st.columns(3)

col1.metric("Baseline Waiting Time", f"{baseline_waiting:.2f} s")

col2.metric(
    "Optimized Waiting Time",
    f"{optimized_waiting:.2f} s",
    delta=f"-{baseline_waiting - optimized_waiting:.2f} s"
)

improvement = ((baseline_waiting - optimized_waiting) / baseline_waiting) * 100
col3.metric("Improvement", f"{improvement:.1f} %")

# ---------------- Charts ----------------
st.subheader("📉 Optimization Results")

col4, col5 = st.columns(2)

# Convergence Curve
with col4:
    fig1, ax1 = plt.subplots()
    ax1.plot(range(1, generations + 1), fitness_history)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness (Waiting Time)")
    ax1.set_title("ES Convergence Curve")
    ax1.grid(True)
    st.pyplot(fig1)

# Before vs After
with col5:
    fig2, ax2 = plt.subplots()
    ax2.bar(
        ["Before Optimization", "After ES"],
        [baseline_waiting, optimized_waiting]
    )
    ax2.set_ylabel("Average Waiting Time (seconds)")
    ax2.set_title("Waiting Time Comparison")
    st.pyplot(fig2)

# ---------------- Dataset Insight ----------------
st.subheader("📈 Traffic Waiting Time Distribution")

fig3, ax3 = plt.subplots()
ax3.hist(df["waiting_time"], bins=20)
ax3.set_xlabel("Waiting Time (seconds)")
ax3.set_ylabel("Frequency")
ax3.set_title("Dataset Waiting Time Distribution")
st.pyplot(fig3)

# ---------------- Explanation ----------------
st.success(
    "Changing ES parameters in the sidebar re-runs the optimization. "
    "Graphs and performance metrics update dynamically based on the algorithm behaviour."
)

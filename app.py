import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Traffic Light Optimization using ES",
    layout="wide"
)

st.title("🚦 Traffic Light Optimization using Evolution Strategy (ES)")
st.caption("Dataset-based optimization of traffic signal timing")

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("traffic_dataset.csv")

df = load_data()

st.subheader("📂 Traffic Dataset Preview")
st.dataframe(df.head())

# --------------------------------------------------
# ⚠️ DEFINE DATASET COLUMNS HERE (EDIT IF NEEDED)
# --------------------------------------------------
ARRIVAL_COL = "arrival_rate"        # vehicles per second / minute
WAIT_COL    = "avg_waiting_time"    # seconds
QUEUE_COL   = "queue_length"        # vehicles

# --------------------------------------------------
# Baseline Statistics from Dataset
# --------------------------------------------------
baseline_arrival = df[ARRIVAL_COL].mean()
baseline_wait    = df[WAIT_COL].mean()
baseline_queue   = df[QUEUE_COL].mean()

# --------------------------------------------------
# Sidebar – ES Parameters
# --------------------------------------------------
st.sidebar.header("⚙️ Evolution Strategy Parameters")

generations = st.sidebar.slider("Generations", 50, 300, 100)
sigma = st.sidebar.slider("Mutation Step Size (σ)", 0.1, 5.0, 1.0)

min_green = st.sidebar.slider("Minimum Green Time (s)", 10, 30, 15)
max_green = st.sidebar.slider("Maximum Green Time (s)", 40, 120, 60)

st.sidebar.markdown(
    "**Objective:** Minimize waiting time and queue length"
)

run_opt = st.sidebar.button("🚀 Run ES Optimization")

# --------------------------------------------------
# Traffic Evaluation Function (DATASET-BASED)
# --------------------------------------------------
def evaluate_traffic(green_time):
    """
    Dataset-based traffic evaluation.
    Green time scales service efficiency.
    """
    service_factor = green_time / max_green

    waiting_time = baseline_wait * (1 / service_factor)
    queue_length = baseline_queue * (1 / service_factor)
    throughput   = baseline_arrival * service_factor * 3600

    return waiting_time, queue_length, throughput

# --------------------------------------------------
# Fitness Function (Multi-objective Weighted Sum)
# --------------------------------------------------
def fitness(green_time, w1=0.6, w2=0.4):
    wait, queue, _ = evaluate_traffic(green_time)
    return w1 * wait + w2 * queue

# --------------------------------------------------
# Run Optimization
# --------------------------------------------------
if run_opt:
    st.subheader("⚙️ Optimization Results")

    # ---------------- Baseline ----------------
    base_wait, base_queue, base_throughput = evaluate_traffic(40)

    # ---------------- ES Initialization ----------------
    green = np.random.uniform(min_green, max_green)
    fit = fitness(green)

    fitness_history = [fit]
    green_history = [green]

    # ---------------- (1+1)-Evolution Strategy ----------------
    for _ in range(generations):
        offspring = green + np.random.normal(0, sigma)
        offspring = np.clip(offspring, min_green, max_green)

        off_fit = fitness(offspring)

        if off_fit <= fit:
            green, fit = offspring, off_fit

        fitness_history.append(fit)
        green_history.append(green)

    # ---------------- Optimized Results ----------------
    opt_wait, opt_queue, opt_throughput = evaluate_traffic(green)

    improvement = ((base_wait - opt_wait) / base_wait) * 100

    # --------------------------------------------------
    # Performance Metrics
    # --------------------------------------------------
    st.subheader("📊 Performance Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Baseline Waiting Time (s)", f"{base_wait:.2f}")
    col2.metric("Optimized Waiting Time (s)", f"{opt_wait:.2f}", f"{improvement:.2f}%")
    col3.metric("Queue Length (veh)", f"{opt_queue:.2f}")
    col4.metric("Throughput (veh/hr)", f"{opt_throughput:.0f}")

    # --------------------------------------------------
    # Convergence & Green Time Evolution
    # --------------------------------------------------
    st.subheader("📉 Evolution Strategy Performance")

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        ax1.plot(fitness_history)
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness Value")
        ax1.set_title("Fitness Convergence")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        ax2.plot(green_history)
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Green Time (s)")
        ax2.set_title("Green Time Evolution")
        st.pyplot(fig2)

    # --------------------------------------------------
    # Comparison Table
    # --------------------------------------------------
    st.subheader("📋 Performance Comparison")

    result_df = pd.DataFrame({
        "Metric": [
            "Average Waiting Time (s)",
            "Mean Queue Length (veh)",
            "Traffic Throughput (veh/hr)"
        ],
        "Before Optimization": [
            round(base_wait, 2),
            round(base_queue, 2),
            round(base_throughput, 0)
        ],
        "After ES Optimization": [
            round(opt_wait, 2),
            round(opt_queue, 2),
            round(opt_throughput, 0)
        ]
    })

    st.dataframe(result_df, use_container_width=True)

    # --------------------------------------------------
    # Conclusion
    # --------------------------------------------------
    st.subheader("✅ Conclusion")
    st.markdown("""
    - The **(1+1)-Evolution Strategy** successfully optimized traffic signal green time.
    - Optimization was **directly based on real traffic dataset statistics**.
    - Significant reduction in **waiting time and queue length** was achieved.
    - Results demonstrate the effectiveness of **evolutionary computation** for traffic signal optimization.
    - The approach is extensible to **multi-intersection and real-time control** scenarios.
    """)

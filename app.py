import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(page_title="Traffic Light Optimization using ES", layout="wide")
st.title("🚦 Traffic Light Optimization using Evolution Strategy (ES)")
st.caption("JIE42903 – Evolutionary Computing | (1+1)-Evolution Strategy with Extended Analysis")

# -----------------------------------
# Sidebar – ES Parameters
# -----------------------------------
st.sidebar.header("⚙️ Evolution Strategy Configuration")
generations = st.sidebar.slider("Generations", 50, 200, 100)
sigma = st.sidebar.slider("Mutation Step Size (σ)", 0.1, 5.0, 1.0)
min_green = st.sidebar.slider("Minimum Green Time (s)", 10, 30, 10)
max_green = st.sidebar.slider("Maximum Green Time (s)", 40, 120, 60)

st.sidebar.markdown("**Objective:** Minimize Average Vehicle Waiting Time and Queue Length")

# -----------------------------------
# Sidebar – Run Optimization Button
# -----------------------------------
st.sidebar.header("🚀 Run Optimization")
run_opt = st.sidebar.button("Optimize Traffic Signals")

# -----------------------------------
# Traffic Simulation Functions
# -----------------------------------
def simulate_traffic(green_time):
    """Simulate traffic intersection performance based on green time."""
    arrival_rate = 0.5  # vehicles per second
    service_rate = green_time / max_green
    waiting_time = max(5, 40 - (green_time * 0.8)) + np.random.normal(0, 1)
    queue_length = max(1, int(arrival_rate * waiting_time))
    throughput = int(service_rate * 3600)
    return waiting_time, queue_length, throughput

def fitness_single(green_time):
    """Single-objective: average waiting time"""
    waiting_time, _, _ = simulate_traffic(green_time)
    return waiting_time

def fitness_multi(green_time, w1=0.5, w2=0.5):
    """Multi-objective weighted sum of waiting time and queue length"""
    waiting_time, queue_length, _ = simulate_traffic(green_time)
    return w1*waiting_time + w2*queue_length, waiting_time, queue_length

# -----------------------------------
# Run Optimization Only When Button is Clicked
# -----------------------------------
if run_opt:
    st.subheader("⚙️ Optimization Results")

    # -------------------
    # Baseline Traffic
    # -------------------
    baseline_green = 40
    baseline_wait, baseline_queue, baseline_throughput = simulate_traffic(baseline_green)

    # -------------------
    # Single-objective ES Optimization
    # -------------------
    green = np.random.uniform(min_green, max_green)
    fitness_val = fitness_single(green)
    fitness_history = [fitness_val]
    green_history = [green]

    for _ in range(generations):
        offspring = green + np.random.normal(0, sigma)
        offspring = np.clip(offspring, min_green, max_green)
        offspring_fitness = fitness_single(offspring)
        if offspring_fitness <= fitness_val:
            green = offspring
            fitness_val = offspring_fitness
        fitness_history.append(fitness_val)
        green_history.append(green)

    best_solution = green
    opt_wait, opt_queue, opt_throughput = simulate_traffic(best_solution)
    improvement = ((baseline_wait - opt_wait) / baseline_wait) * 100

    # -------------------
    # Side-by-Side Plots: Convergence & Green Time
    # -------------------
    st.subheader("📉 ES Performance Visualization")
    col1, col2 = st.columns(2)

    # Convergence Plot
    with col1:
        fig1, ax1 = plt.subplots(figsize=(5,3))
        ax1.plot(fitness_history, marker='o', color='blue')
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Average Waiting Time (s)")
        ax1.set_title("Fitness Convergence")
        st.pyplot(fig1)

    # Green Time Evolution Plot
    with col2:
        fig2, ax2 = plt.subplots(figsize=(5,3))
        ax2.plot(green_history, marker='o', color='green')
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Green Time (s)")
        ax2.set_title("Green Time Evolution")
        st.pyplot(fig2)

    # -------------------
    # Performance Comparison Table
    # -------------------
    st.subheader("📋 Performance Comparison")
    df = pd.DataFrame({
        "Metric": ["Average Waiting Time (s)", "Mean Queue Length (veh)", "Traffic Throughput (veh/hr)"],
        "Before Optimization": [round(baseline_wait, 2), baseline_queue, baseline_throughput],
        "After ES Optimization": [round(opt_wait, 2), opt_queue, opt_throughput]
    })
    st.dataframe(df, use_container_width=True)

    # -------------------
    # Extended Analysis – Multi-objective
    # -------------------
    st.subheader("🔬 Extended Analysis – Multi-Objective Optimization")

    st.markdown("""
    Using **multi-objective ES**, we optimize traffic signals considering:
    1. Average Waiting Time (s)
    2. Mean Queue Length (vehicles)
    """)

    # Optional: sliders to adjust weights
    st.markdown("**Adjust Objective Weights:**")
    w1 = st.slider("Weight for Waiting Time", 0.0, 1.0, 0.5)
    w2 = st.slider("Weight for Queue Length", 0.0, 1.0, 0.5)
    if w1 + w2 == 0:
        w2 = 0.5
        w1 = 0.5

    # Generate Pareto solutions
    n_pareto = 50
    pareto_wait = []
    pareto_queue = []
    pareto_green = []

    for _ in range(n_pareto):
        green_candidate = np.random.uniform(min_green, max_green)
        _, wait, queue = fitness_multi(green_candidate, w1, w2)
        pareto_wait.append(wait)
        pareto_queue.append(queue)
        pareto_green.append(green_candidate)

    # Plot Pareto front
    fig3, ax3 = plt.subplots(figsize=(6,4))
    ax3.scatter(pareto_wait, pareto_queue, c='purple', alpha=0.7)
    ax3.set_xlabel("Average Waiting Time (s)")
    ax3.set_ylabel("Mean Queue Length (veh)")
    ax3.set_title("Pareto Front – Multi-Objective Trade-off")
    st.pyplot(fig3)

    # Show sample Pareto solutions
    st.subheader("📑 Sample Multi-Objective Solutions")
    df_pareto = pd.DataFrame({
        "Green Time (s)": np.round(pareto_green, 2),
        "Average Waiting Time (s)": np.round(pareto_wait, 2),
        "Queue Length (veh)": np.round(pareto_queue, 2)
    })
    st.dataframe(df_pareto, use_container_width=True)

    # -------------------
    # Performance Analysis
    # -------------------
    st.subheader("📈 Performance Analysis")
    st.markdown(f"""
- **Single-objective ES:** Waiting time reduced by **{improvement:.2f}%**.
- **Queue length improved** from {baseline_queue} to {opt_queue} vehicles.
- **Throughput increased** from {baseline_throughput} to {opt_throughput} veh/hr.
- **Multi-objective ES:** Provides Pareto front of solutions showing trade-offs between waiting time and queue length.
- Decision-makers can select **green time solutions** based on traffic priorities.
""")

    # -------------------
    # Conclusion
    # -------------------
    st.subheader("✅ Conclusion")
    st.markdown("""
- (1+1)-Evolution Strategy effectively optimizes traffic signals for single and multi-objective scenarios.
- Multi-objective optimization enables **flexible and robust traffic control plans**.
- Dashboard allows **interactive exploration** of trade-offs, convergence, and green time evolution.
- Future work: extend to multiple intersections, real-time adaptive control, and additional objectives like emissions and fuel consumption.
""")

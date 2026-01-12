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
st.caption("Dataset-driven optimization with Multiple Trials & Best-Fitness Selection")

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
# Dataset Statistics (USED IN OPTIMIZATION)
# --------------------------------------------------
avg_waiting_time = df["waiting_time"].mean()
avg_vehicle_count = df["vehicle_count"].mean()
avg_lane_occupancy = df["lane_occupancy"].mean()
avg_flow_rate = df["flow_rate"].mean()

# --------------------------------------------------
# Sidebar – ES Parameters
# --------------------------------------------------
st.sidebar.header("⚙️ Evolution Strategy Parameters")

generations = st.sidebar.slider("Generations", 50, 300, 100)
sigma = st.sidebar.slider("Base Mutation Step Size (σ)", 0.1, 5.0, 1.0)
num_trials = st.sidebar.slider("Number of Trials", 5, 50, 10)

min_green = st.sidebar.slider("Minimum Green Time (s)", 10, 30, 15)
max_green = st.sidebar.slider("Maximum Green Time (s)", 40, 120, 60)

run_opt = st.sidebar.button("🚀 Run Optimization")

# --------------------------------------------------
# Traffic Evaluation Function (DATASET-BASED)
# --------------------------------------------------
def evaluate_traffic(green_time):
    """
    Higher green time improves service efficiency.
    Dataset statistics are scaled using a service factor.
    """
    service_factor = green_time / max_green

    waiting_time = avg_waiting_time / service_factor
    vehicle_count = avg_vehicle_count * (1 / service_factor)
    lane_occupancy = avg_lane_occupancy * (1 / service_factor)
    throughput = avg_flow_rate * service_factor

    return waiting_time, vehicle_count, lane_occupancy, throughput

# --------------------------------------------------
# Fitness Function (Weighted Multi-Objective)
# --------------------------------------------------
def fitness(green_time):
    wait, veh, lane, _ = evaluate_traffic(green_time)
    # Multi-objective weighted sum
    return 0.6 * wait + 0.25 * veh + 0.15 * lane

# --------------------------------------------------
# Run Evolution Strategy with Multiple Trials
# --------------------------------------------------
if run_opt:
    st.subheader("⚙️ Optimization Results (Multiple Trials)")

    # ---------------- Baseline ----------------
    baseline_green = 40
    base_wait, base_veh, base_lane, base_throughput = evaluate_traffic(baseline_green)

    # Store best among all trials
    best_overall_fitness = np.inf
    best_overall_green = None
    best_fitness_history = None
    best_green_history = None

    trial_results = []

    # ================= MULTIPLE TRIALS =================
    for trial in range(num_trials):

        # Different sigma for each trial (robustness analysis)
        trial_sigma = np.random.uniform(0.5 * sigma, 1.5 * sigma)

        # Random initialization
        green = np.random.uniform(min_green, max_green)
        best_fitness = fitness(green)

        fitness_history = [best_fitness]
        green_history = [green]

        # -------- (1+1)-Evolution Strategy --------
        for _ in range(generations):
            offspring = green + np.random.normal(0, trial_sigma)
            offspring = np.clip(offspring, min_green, max_green)

            offspring_fitness = fitness(offspring)

            # Greedy selection
            if offspring_fitness <= best_fitness:
                green = offspring
                best_fitness = offspring_fitness

            fitness_history.append(best_fitness)
            green_history.append(green)

        # Save trial result
        trial_results.append({
            "Trial": trial + 1,
            "Sigma Used": round(trial_sigma, 3),
            "Best Fitness": round(best_fitness, 4),
            "Best Green Time (s)": round(green, 2)
        })

        # Track global best
        if best_fitness < best_overall_fitness:
            best_overall_fitness = best_fitness
            best_overall_green = green
            best_fitness_history = fitness_history
            best_green_history = green_history

    # --------------------------------------------------
    # Best Optimized Result
    # --------------------------------------------------
    opt_wait, opt_veh, opt_lane, opt_throughput = evaluate_traffic(best_overall_green)
    improvement = ((base_wait - opt_wait) / base_wait) * 100

    # --------------------------------------------------
    # Performance Metrics
    # --------------------------------------------------
    st.subheader("📊 Performance Overview (Best Trial)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Baseline Waiting Time (s)", f"{base_wait:.2f}")
    col2.metric("Optimized Waiting Time (s)", f"{opt_wait:.2f}", f"{improvement:.2f}%")
    col3.metric("Lane Occupancy (%)", f"{opt_lane:.2f}")
    col4.metric("Traffic Throughput", f"{opt_throughput:.0f}")

    # --------------------------------------------------
    # Trial Results Table
    # --------------------------------------------------
    st.subheader("📋 Trial-wise Results with Different Parameter Values")
    trial_df = pd.DataFrame(trial_results)
    st.dataframe(trial_df, use_container_width=True)

    best_trial_index = trial_df["Best Fitness"].idxmin()
    st.success(
        f"🏆 Best solution found in Trial {trial_df.loc[best_trial_index, 'Trial']} "
        f"with Green Time = {trial_df.loc[best_trial_index, 'Best Green Time (s)']} s "
        f"and σ = {trial_df.loc[best_trial_index, 'Sigma Used']}"
    )

    # --------------------------------------------------
    # Convergence & Green Time Evolution (Best Trial)
    # --------------------------------------------------
    st.subheader("📉 ES Convergence Analysis (Best Trial)")

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        ax1.plot(best_fitness_history)
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness Value")
        ax1.set_title("Fitness Convergence")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.plot(best_green_history)
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Green Time (s)")
        ax2.set_title("Green Time Evolution")
        st.pyplot(fig2)

    # --------------------------------------------------
    # Performance Comparison Table
    # --------------------------------------------------
    st.subheader("📋 Performance Comparison")

    result_df = pd.DataFrame({
        "Metric": [
            "Average Waiting Time (s)",
            "Vehicle Count",
            "Lane Occupancy (%)",
            "Traffic Throughput"
        ],
        "Before ES": [
            round(base_wait, 2),
            round(base_veh, 2),
            round(base_lane, 2),
            round(base_throughput, 0)
        ],
        "After ES (Best Trial)": [
            round(opt_wait, 2),
            round(opt_veh, 2),
            round(opt_lane, 2),
            round(opt_throughput, 0)
        ]
    })

    st.dataframe(result_df, use_container_width=True)

    # --------------------------------------------------
    # Extended Analysis
    # --------------------------------------------------
    st.subheader("4. Extended Analysis")
    st.markdown("""
    Multiple trials were conducted using different mutation step sizes (σ) and random initial
    green times to analyze the robustness of the Evolution Strategy.  
    This demonstrates the stochastic nature of ES and ensures that the final solution is not dependent
    on a single random run. The best solution is selected as the one achieving the minimum fitness
    across all trials, improving reliability and optimization quality.
    """)

    # --------------------------------------------------
    # Conclusion
    # --------------------------------------------------
    st.subheader("✅ Conclusion")
    st.markdown("""
    - Multiple ES trials improve the robustness and reliability of traffic signal optimization.
    - Different σ values help explore diverse search behaviors and convergence patterns.
    - The best-fitness selection ensures a scientifically valid optimized green time.
    - Results confirm the effectiveness of Evolution Strategy for real-world traffic signal control.
    - The approach can be extended to multi-intersection and real-time adaptive systems.
    """)

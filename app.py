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
st.caption("Dataset-driven optimization with Multiple Trials & Best-Fitness Analysis")

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
sigma = st.sidebar.slider("Mutation Step Size (σ)", 0.1, 5.0, 1.0)
num_trials = st.sidebar.slider("Number of Trials", 5, 50, 10)

min_green = st.sidebar.slider("Minimum Green Time (s)", 10, 30, 15)
max_green = st.sidebar.slider("Maximum Green Time (s)", 40, 120, 60)

run_opt = st.sidebar.button("🚀 Run Optimization")

# --------------------------------------------------
# Traffic Evaluation Function (DATASET-BASED)
# --------------------------------------------------
def evaluate_traffic(green_time):
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
    return 0.6 * wait + 0.25 * veh + 0.15 * lane

# --------------------------------------------------
# Run Evolution Strategy with Multiple Trials
# --------------------------------------------------
if run_opt:
    st.subheader("⚙️ Optimization Results (Multiple Trials)")

    # ---------------- Baseline ----------------
    baseline_green = 40
    base_wait, base_veh, base_lane, base_throughput = evaluate_traffic(baseline_green)

    best_overall_fitness = np.inf
    best_overall_green = None
    best_fitness_history = None
    best_green_history = None

    trial_results = []

    # ================= MULTIPLE TRIALS =================
    for trial in range(num_trials):

        green = np.random.uniform(min_green, max_green)
        best_fitness = fitness(green)

        fitness_history = [best_fitness]
        green_history = [green]

        for _ in range(generations):
            offspring = green + np.random.normal(0, sigma)
            offspring = np.clip(offspring, min_green, max_green)

            offspring_fitness = fitness(offspring)

            if offspring_fitness <= best_fitness:
                green = offspring
                best_fitness = offspring_fitness

            fitness_history.append(best_fitness)
            green_history.append(green)

        trial_results.append((trial+1, best_fitness, green))

        # Save global best among all trials
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
    st.subheader("📋 Trial-wise Best Fitness Results")

    trial_df = pd.DataFrame(trial_results, columns=["Trial", "Best Fitness", "Best Green Time (s)"])
    st.dataframe(trial_df, use_container_width=True)

    st.success(f"🏆 Best solution found in Trial {trial_df['Best Fitness'].idxmin()+1} "
               f"with Green Time = {best_overall_green:.2f}s")

    # --------------------------------------------------
    # Convergence & Green Time Evolution (Best Trial)
    # --------------------------------------------------
    st.subheader("📉 ES Convergence Analysis (Best Trial)")

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(5,3))
        ax1.plot(best_fitness_history)
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness Value")
        ax1.set_title("Fitness Convergence")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(5,3))
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
    Multiple trials demonstrate the **stochastic nature** of Evolution Strategy.  
    Different runs explore different regions of the solution space, producing varied results.  
    Selecting the **best fitness among all trials** ensures a robust and reliable optimized signal timing plan.
    """)

    # --------------------------------------------------
    # Conclusion
    # --------------------------------------------------
    st.subheader("✅ Conclusion")
    st.markdown("""
    - Multiple ES trials improve reliability and robustness of optimization results.
    - The best solution is selected based on **global minimum fitness**.
    - Results show clear reduction in waiting time and congestion indicators.
    - This validates the suitability of **Evolution Strategy** for traffic signal optimization.
    - The approach can be extended to **multi-objective and real-time systems**.
    """)

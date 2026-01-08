import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(page_title="Traffic Light Optimization using ES", layout="wide")

st.title("🚦 Traffic Light Optimization using Evolution Strategy (ES)")
st.caption("JIE42903 – Evolutionary Computing | (1+1)-Evolution Strategy")

# -----------------------------------
# Sidebar – ES Configuration
# -----------------------------------
st.sidebar.header("⚙️ Evolution Strategy Configuration")

generations = st.sidebar.slider("Generations", 50, 200, 100)
sigma = st.sidebar.slider("Mutation Strength (σ)", 0.1, 5.0, 1.0)

min_green = st.sidebar.slider("Min Green Time (s)", 10, 30, 10)
max_green = st.sidebar.slider("Max Green Time (s)", 40, 120, 60)

st.sidebar.markdown("**Objective:** Minimize Average Waiting Time")

# -----------------------------------
# Traffic Simulation Functions
# -----------------------------------
def simulate_traffic(green_time):
    """
    Simulates traffic performance based on green time
    """
    arrival_rate = 0.5  # vehicles per second
    service_rate = green_time / max_green

    waiting_time = max(5, 40 - (green_time * 0.8)) + np.random.normal(0, 1)
    queue_length = max(1, int(arrival_rate * waiting_time))
    throughput = int(service_rate * 3600)

    return waiting_time, queue_length, throughput


def fitness(green_time):
    waiting_time, _, _ = simulate_traffic(green_time)
    return waiting_time


# -----------------------------------
# (1+1)-Evolution Strategy
# -----------------------------------
green = np.random.uniform(min_green, max_green)
fitness_val = fitness(green)

fitness_history = []
green_history = []

for _ in range(generations):
    offspring = green + np.random.normal(0, sigma)
    offspring = np.clip(offspring, min_green, max_green)

    offspring_fitness = fitness(offspring)

    if offspring_fitness < fitness_val:
        green = offspring
        fitness_val = offspring_fitness

    fitness_history.append(fitness_val)
    green_history.append(green)

# -----------------------------------
# Baseline vs Optimized Performance
# -----------------------------------
baseline_green = 40
baseline_wait, baseline_queue, baseline_throughput = simulate_traffic(baseline_green)

opt_wait, opt_queue, opt_throughput = simulate_traffic(green)

improvement = ((baseline_wait - opt_wait) / baseline_wait) * 100

# -----------------------------------
# Dashboard Metrics
# -----------------------------------
st.subheader("📊 Performance Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Baseline Waiting Time (s)", f"{baseline_wait:.2f}")
col2.metric("Optimized Waiting Time (s)", f"{opt_wait:.2f}", f"{improvement:.2f}%")
col3.metric("Mean Queue Length (veh)", f"{opt_queue}")
col4.metric("Throughput (veh/hr)", f"{opt_throughput}")

# -----------------------------------
# Fitness Convergence Plot
# -----------------------------------
st.subheader("📉 ES Fitness Convergence")

fig1 = plt.figure()
plt.plot(fitness_history)
plt.xlabel("Generation")
plt.ylabel("Average Waiting Time (s)")
plt.title("Fitness Convergence")
st.pyplot(fig1)

# -----------------------------------
# Green Time Evolution Plot
# -----------------------------------
st.subheader("🧬 Green Time Evolution")

fig2 = plt.figure()
plt.plot(green_history)
plt.xlabel("Generation")
plt.ylabel("Green Time (s)")
plt.title("Evolution of Green Time")
st.pyplot(fig2)

# -----------------------------------
# Performance Comparison Table
# -----------------------------------
st.subheader("📋 Performance Comparison")

df = pd.DataFrame({
    "Metric": [
        "Average Waiting Time (s)",
        "Mean Queue Length (veh)",
        "Traffic Throughput (veh/hr)"
    ],
    "Before Optimization": [
        round(baseline_wait, 2),
        baseline_queue,
        baseline_throughput
    ],
    "After ES Optimization": [
        round(opt_wait, 2),
        opt_queue,
        opt_throughput
    ]
})

st.dataframe(df, use_container_width=True)

# -----------------------------------
# Conclusion
# -----------------------------------
st.subheader("✅ Conclusion")

st.markdown("""
- Evolution Strategy effectively reduces average vehicle waiting time.
- Queue length and traffic throughput improve after optimization.
- ES shows stable convergence and is computationally efficient.
- Suitable for traffic signal optimization problems.

**Future Work:**
- Multi-intersection optimization  
- Real-time traffic data  
- Multi-objective ES (delay, emissions, fuel)
""")

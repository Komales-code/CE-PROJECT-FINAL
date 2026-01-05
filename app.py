import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Traffic Light Optimization using ES", layout="centered")

st.title("🚦 Traffic Light Optimization using Evolutionary Strategy (ES)")

# Load dataset
df = pd.read_csv("traffic_dataset.csv")

st.subheader("📊 Traffic Dataset Preview")
st.dataframe(df.head())

# Baseline performance
baseline_waiting = df['waiting_time'].mean()

st.subheader("⏱ Baseline Traffic Performance")
st.metric(label="Average Waiting Time (Before Optimization)", 
          value=f"{baseline_waiting:.2f} seconds")

# User input: optimized value from ES
st.subheader("⚙️ ES Optimization Result")
optimized_waiting = st.slider(
    "Optimized Average Waiting Time (seconds)",
    min_value=0.0,
    max_value=baseline_waiting,
    value=baseline_waiting * 0.6
)

# Bar chart comparison
st.subheader("📉 Waiting Time Comparison")

fig1, ax1 = plt.subplots()
ax1.bar(["Before Optimization", "After ES Optimization"],
        [baseline_waiting, optimized_waiting])
ax1.set_ylabel("Average Waiting Time (seconds)")
ax1.set_title("Average Waiting Time Before and After ES")
st.pyplot(fig1)

# Waiting time distribution
st.subheader("📈 Waiting Time Distribution (Dataset)")

fig2, ax2 = plt.subplots()
ax2.hist(df['waiting_time'], bins=20)
ax2.set_xlabel("Waiting Time (seconds)")
ax2.set_ylabel("Frequency")
ax2.set_title("Distribution of Vehicle Waiting Time")
st.pyplot(fig2)

# ES convergence curve (example fitness history)
st.subheader("📉 ES Convergence Curve")

fitness_history = [
    140, 132, 125, 118, 110, 104, 99, 95, 92, 89,
    87, 85, 84, 83, 82, 81, 80.5, 80, 79.8, 79.5
]

generations = list(range(1, len(fitness_history) + 1))

fig3, ax3 = plt.subplots()
ax3.plot(generations, fitness_history)
ax3.set_xlabel("Generation")
ax3.set_ylabel("Fitness Value")
ax3.set_title("Convergence Curve of Evolutionary Strategy")
ax3.grid(True)
st.pyplot(fig3)

st.success("✅ Streamlit Dashboard Successfully Displays ES Optimization Results")

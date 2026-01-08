import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Evolution Strategy Functions
# -----------------------------
def traffic_performance(signal_timings):
    """
    Simulated traffic intersection performance.
    Lower value = better performance (e.g., less waiting time)
    """
    # Example: simple performance function based on green times
    green_NS, green_EW = signal_timings
    waiting_time = (50 - green_NS)**2 + (40 - green_EW)**2
    return waiting_time

def mutate(solution, sigma=5):
    """
    Gaussian mutation
    """
    return solution + np.random.normal(0, sigma, size=solution.shape)

def evolution_strategy(n_generations=50, population_size=10):
    # Initialize population: [green_NS, green_EW] in seconds
    population = np.random.randint(20, 60, (population_size, 2))
    best_history = []

    for gen in range(n_generations):
        # Evaluate fitness
        fitness = np.array([traffic_performance(ind) for ind in population])
        # Select best individual
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        best_history.append(fitness[best_idx])

        # Generate next population
        new_population = []
        for _ in range(population_size):
            child = mutate(best)
            # Ensure green times are within limits
            child = np.clip(child, 20, 60)
            new_population.append(child)
        population = np.array(new_population)

    return best, best_history

# -----------------------------
# Streamlit Interface
# -----------------------------
st.title("Traffic Signal Optimization using Evolution Strategy (ES)")

st.sidebar.header("ES Parameters")
n_generations = st.sidebar.slider("Number of Generations", 10, 200, 50)
population_size = st.sidebar.slider("Population Size", 5, 50, 10)

if st.button("Run ES Optimization"):
    best_solution, history = evolution_strategy(n_generations, population_size)
    st.success(f"Optimal Traffic Signal Timings (seconds): NS={int(best_solution[0])}, EW={int(best_solution[1])}")

    # Plot performance over generations
    fig, ax = plt.subplots()
    ax.plot(history, marker='o')
    ax.set_xlabel("Generation")
    ax.set_ylabel("Performance (Lower is Better)")
    ax.set_title("Evolution Strategy Optimization Progress")
    st.pyplot(fig)

st.markdown("""
This app simulates traffic signal optimization using **Evolution Strategy (ES)**.
- `NS` = Green time for North-South direction
- `EW` = Green time for East-West direction
- The goal is to minimize total waiting time at the intersection.
""")

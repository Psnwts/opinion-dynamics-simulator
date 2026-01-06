import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from models.base_model import deffuant_simulation
from models.convinced_model import convinced_model
from models.polarized_model import polarized_model
from models.influencer_model import influencer_model

st.set_page_config(page_title="Opinion Dynamics", page_icon="🗣️", layout="wide")

st.title("🗣️ Opinion Dynamics Simulator")
st.markdown("Agent-based modeling of opinion formation in social networks")

# Sidebar - Model selection
st.sidebar.header("⚙️ Configuration")
model_choice = st.sidebar.selectbox(
    "Select Model:",
    ["Base (Deffuant)", "Convinced", "Polarized", "Influencer"]
)

# Common parameters
st.sidebar.subheader("Population")
pop_size = st.sidebar.slider("Population size:", 50, 500, 100)
iterations = st.sidebar.slider("Iterations:", 100, 2000, 500)

# Model-specific parameters
st.sidebar.subheader("Model Parameters")

if model_choice == "Base (Deffuant)":
    epsilon = st.sidebar.slider("Epsilon (threshold):", 0.0, 1.0, 0.2, 0.05)
    
elif model_choice == "Convinced":
    epsilon = st.sidebar.slider("Epsilon (threshold):", 0.0, 1.0, 0.2, 0.05)
    bornesup = st.sidebar.slider("Conviction bound:", 0.0, 1.0, 0.8, 0.05)
    distmax = st.sidebar.slider("Distance for conviction increase:", 0.0, 0.5, 0.09, 0.01)
    
elif model_choice == "Polarized":
    weight = st.sidebar.slider("Extremist influence weight:", 1.0, 10.0, 1.0, 0.5)
    center = st.sidebar.slider("Center point:", 0.0, 1.0, 0.5, 0.05)
    
else:  # Influencer
    num_influencers = st.sidebar.number_input("Number of influencers:", 1, 5, 2)
    allow_competition = st.sidebar.checkbox("Allow competition", value=False)
    
    st.sidebar.write("**Influencer Settings:**")
    inf_opinions = []
    inf_weights = []
    for i in range(num_influencers):
        col1, col2 = st.sidebar.columns(2)
        with col1:
            op = st.number_input(f"Inf {i+1} opinion:", 0.0, 1.0, 0.1 if i == 0 else 0.8, 0.1, key=f"op{i}")
        with col2:
            wt = st.number_input(f"Weight:", 2.0, 10.0, 10.0 if i == 0 else 8.0, 1.0, key=f"wt{i}")
        inf_opinions.append(op)
        inf_weights.append(wt)

# Run simulation
if st.button("▶️ Run Simulation", type="primary", use_container_width=True):
    
    with st.spinner("Running simulation..."):
        # Run model
        if model_choice == "Base (Deffuant)":
            history = deffuant_simulation(pop_size, epsilon, iterations)
            
        elif model_choice == "Convinced":
            history, conviction_history = convinced_model(pop_size, epsilon, bornesup, distmax, iterations)
            
        elif model_choice == "Polarized":
            history = polarized_model(pop_size, weight, center, iterations)
            
        else:  # Influencer
            history, population = influencer_model(
                pop_size, num_influencers, iterations,
                inf_opinions, inf_weights, allow_competition
            )
    
    st.success("✅ Simulation complete!")
    
    # Results
    st.header("📊 Results")
    
    # Metrics
    final_opinions = history[-1]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Final Variance", f"{np.var(final_opinions):.4f}")
    with col2:
        unique = len(np.unique(np.round(final_opinions, 2)))
        st.metric("Opinion Clusters", unique)
    with col3:
        st.metric("Opinion Range", f"[{final_opinions.min():.2f}, {final_opinions.max():.2f}]")
    with col4:
        variance_change = np.abs(np.diff(np.var(history, axis=1)))
        converged = "Yes" if variance_change[-10:].mean() < 0.001 else "No"
        st.metric("Converged", converged)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Opinion Evolution")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot trajectories
        for agent in range(pop_size):
            ax.plot(history[:, agent], alpha=0.3, linewidth=0.5)
        
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Opinion")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_title("Opinion Trajectories")
        st.pyplot(fig)
    
    with col2:
        st.subheader("📊 Final Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(final_opinions, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel("Opinion")
        ax.set_ylabel("Frequency")
        ax.set_xlim(-0.05, 1.05)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_title("Final Opinion Distribution")
        st.pyplot(fig)
    
    # Variance over time
    st.subheader("📉 Convergence Analysis")
    fig, ax = plt.subplots(figsize=(12, 4))
    
    variance_over_time = np.var(history, axis=1)
    ax.plot(variance_over_time, color='red', linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Opinion Variance")
    ax.grid(True, alpha=0.3)
    ax.set_title("Opinion Variance Over Time")
    st.pyplot(fig)
    
    # Conviction history (if convinced model)
    if model_choice == "Convinced":
        st.subheader("💪 Conviction Evolution")
        fig, ax = plt.subplots(figsize=(12, 4))
        
        for agent in range(min(pop_size, 50)):  # Show first 50
            ax.plot(conviction_history[:, agent], alpha=0.3, linewidth=0.5)
        
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Conviction")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_title("Conviction Trajectories (first 50 agents)")
        st.pyplot(fig)

# Footer
st.divider()
st.caption("🗣️ Opinion Dynamics Simulator | Agent-Based Modeling")
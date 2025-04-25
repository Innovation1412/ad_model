import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ODE system
def odefun(t, y, p):
    S, B, G = y
    kinetics = p['kinetics']

    if kinetics == 'monod':
        mu = p['mu_max'] * S / (p['K_S'] + S)
        R_BS = mu * B
    elif kinetics == 'linear':
        R_BS = p['k'] * S
    elif kinetics == 'haldane':
        mu = p['mu_max'] * S / (p['K_S'] + S + (S**2 / p['K_I']))
        R_BS = mu * B
    elif kinetics == 'contois':
        mu = p['mu_max'] * S / (p['K_C'] * B + S)
        R_BS = mu * B
    elif kinetics == 'teissier':
        mu = p['mu_max'] * (1 - np.exp(-S / p['K_T']))
        R_BS = mu * B
    else:
        raise ValueError(f"Unknown kinetics: {kinetics}")

    R_sub = R_BS / p['Y_b']
    R_gas = p['Y_g'] * R_sub

    dS = -R_sub
    dB = R_BS
    dG = R_gas

    return [dS, dB, dG]

# Streamlit UI
st.title("Anaerobic Digestion Model")

with st.sidebar:
    st.header("Model Inputs")
    S0 = st.number_input("Initial Substrate [g/L]", value=100.0)
    B0 = st.number_input("Initial Biomass [g/L]", value=1.0)
    T_end = st.slider("Simulation Time [days]", min_value=10, max_value=100, value=50)

    st.subheader("Kinetic Parameters")
    mu_max = st.number_input("mu_max [1/day]", value=0.4)
    K_S = st.number_input("K_S [g/L]", value=20.0)
    K_I = st.number_input("K_I [g²/L²]", value=250.0)
    K_C = st.number_input("K_C [L/g]", value=5.0)
    K_T = st.number_input("K_T [g/L]", value=15.0)
    k = st.number_input("k [1/day]", value=0.05)
    Y_b = st.number_input("Biomass yield Y_b [g/g]", value=0.3)

    kinetics = st.selectbox("Kinetics type", ["monod", "linear", "haldane", "contois", "teissier"])

# Run simulation
if st.button("Run Simulation"):
    p = {
        'mu_max': mu_max,
        'K_S': K_S,
        'K_I': K_I,
        'K_C': K_C,
        'K_T': K_T,
        'k': k,
        'Y_b': Y_b,
        'Y_g': 1 - Y_b,
        'kinetics': kinetics
    }

    y0 = [S0, B0, 0.0]
    t_span = [0, T_end]
    sol = solve_ivp(lambda t, y: odefun(t, y, p), t_span, y0, dense_output=True)

    t = np.linspace(0, T_end, 300)
    y = sol.sol(t)

    # Plotting
    fig, ax = plt.subplots()
    ax.plot(t, y[0], label='Substrate [g/L]', color='blue')
    ax.plot(t, y[1], label='Biomass [g/L]', color='red')
    ax.plot(t, y[2], label='Biogas [g/L]', color='green')
    ax.set_title("Anaerobic Digestion Simulation")
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Concentration [g/L]")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

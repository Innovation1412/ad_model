import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def odefun(t, y, p):
    S_j, B, G = y

    kinetics = p['kinetics']
    mu_max = p['mu_max']
    if kinetics == 'monod':
        mu = mu_max * S_j / (p['K_S'] + S_j)
        R_BS_j = mu * B
    elif kinetics == 'linear':
        R_BS_j = p['k'] * S_j
    elif kinetics == 'haldane':
        mu = mu_max * (S_j / (p['K_S'] + S_j)) * (p['K_I'] / (p['K_I'] + S_j))
        R_BS_j = mu * B
    elif kinetics == 'contois':
        mu = mu_max * ((S_j / B) / (p['K_C'] + S_j / B))
        R_BS_j = mu * B
    elif kinetics == 'teissier':
        mu = mu_max * (1 - np.exp(-S_j / p['K_T']))
        R_BS_j = mu * B
    elif kinetics == 'moser':
        mu = mu_max * S_j**p['n'] / (p['K_S'] + S_j**p['n'])
        R_BS_j = mu * B
    elif kinetics == 'chen-hashimoto':
        S_ratio = S_j / p['S0']
        mu = mu_max * S_ratio / (p['k_CH'] + S_ratio * (1 - S_ratio))
        R_BS_j = mu * B
    elif kinetics == 'andrews':
        mu = mu_max * S_j / (p['K_S'] + S_j + S_j**2 / p['K_I'])
        R_BS_j = mu * B
    elif kinetics == 'ierusalimsky':
        mu = mu_max * (S_j / (p['K_S'] + S_j)) * (p['K_P'] / (p['K_P'] + S_j))
        R_BS_j = mu * B
    else:
        raise ValueError(f"Unknown kinetics type: {kinetics}")

    R_sub = R_BS_j / p['Y_b']
    R_gas = p['Y_g'] * R_sub

    dS = -R_sub
    dB = R_BS_j
    dG = R_gas
    return [dS, dB, dG]

# Streamlit App
st.title("Anaerobic Digestion Model (ADModelApp)")

with st.sidebar:
    st.header("Initial Conditions")
    S0 = st.number_input("Substrate [g/L]", value=100.0)
    B0 = st.number_input("Biomass [g/L]", value=1.0)
    T_end = st.slider("Time [days]", 10, 100, 50)

    st.header("Model Parameters")
    mu_max = st.number_input("mu_max", value=0.4)
    K_S = st.number_input("K_S", value=20.0)
    K_I = st.number_input("K_I", value=250.0)
    K_C = st.number_input("K_C", value=3.5)
    K_T = st.number_input("K_T", value=15.0)
    k = st.number_input("k (linear rate)", value=0.05)
    Y_b = st.number_input("Y_b (biomass yield)", value=0.3)
    kinetics = st.selectbox("Kinetics", [
        'monod', 'linear', 'haldane', 'contois', 'teissier',
        'moser', 'chen-hashimoto', 'andrews', 'ierusalimsky'
    ])
    n = st.number_input("Moser: n", value=1.5)
    S0_moser = st.number_input("S0 (Chen-H/Ref S)", value=100.0)
    k_CH = st.number_input("k_CH", value=0.2)
    K_P = st.number_input("K_P", value=300.0)

if st.button("Run Simulation"):
    y0 = [S0, B0, 0.0]
    params = {
        'mu_max': mu_max, 'K_S': K_S, 'K_I': K_I, 'K_C': K_C,
        'K_T': K_T, 'k': k, 'Y_b': Y_b, 'Y_g': 1 - Y_b, 'kinetics': kinetics,
        'n': n, 'S0': S0_moser, 'k_CH': k_CH, 'K_P': K_P
    }

    sol = solve_ivp(lambda t, y: odefun(t, y, params), [0, T_end], y0, t_eval=np.linspace(0, T_end, 300))
    t = sol.t
    S = sol.y[0]
    B = sol.y[1]
    G = sol.y[2]
    digestate = S + B

    fig, ax = plt.subplots()
    ax.plot(t, S, 'b', label='Substrate')
    ax.plot(t, B, 'r', label='Biomass')
    ax.plot(t, G, 'g', label='Biogas')
    ax.plot(t, digestate, 'k--', label='Digestate')
    ax.set_title("Anaerobic Digestion Model")
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Concentration [g/L]")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

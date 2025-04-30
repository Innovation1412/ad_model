import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

st.set_page_config(page_title="Anaerobic Digestion Model", layout="wide")
st.title("Anaerobic Digestion Simulator")

# Sidebar Controls
st.sidebar.subheader("1️⃣ Choose Kinetics")
kinetic_descriptions = {
    "monod": r"\mu = \mu_{\text{max}} \cdot \frac{S}{K_S + S}",
    "linear": r"R = k \cdot S",
    "haldane": r"\mu = \mu_{\text{max}} \cdot \frac{S}{K_S + S} \cdot \frac{K_I}{K_I + S}",
    "contois": r"\mu = \mu_{\text{max}} \cdot \frac{S/B}{K_C + S/B}",
    "teissier": r"\mu = \mu_{\text{max}} \cdot (1 - e^{-S/K_T})",
    "moser": r"\mu = \mu_{\text{max}} \cdot \frac{S^n}{K_S + S^n}",
    "chen-hashimoto": r"\mu = \mu_{\text{max}} \cdot \frac{S/S_0}{k_{CH} + \frac{S}{S_0}(1 - \frac{S}{S_0})}",
    "andrews": r"\mu = \mu_{\text{max}} \cdot \frac{S}{K_S + S + \frac{S^2}{K_I}}",
    "ierusalimsky": r"\mu = \mu_{\text{max}} \cdot \frac{S}{K_S + S} \cdot \frac{K_P}{K_P + S}",
}
kinetics = st.sidebar.selectbox("Kinetic model", list(kinetic_descriptions.keys()))
st.sidebar.latex(kinetic_descriptions[kinetics])

params = {'kinetics': kinetics}

st.sidebar.subheader("2️⃣ Kinetic Parameters")
if kinetics in ['monod', 'haldane', 'contois', 'teissier', 'moser', 'chen-hashimoto', 'andrews', 'ierusalimsky']:
    params['mu_max'] = st.sidebar.number_input("μ_max [1/day]", 0.01, 5.0, 0.4)
if kinetics in ['monod', 'haldane', 'moser', 'andrews', 'ierusalimsky']:
    params['K_S'] = st.sidebar.number_input("K_S [g/L]", 0.1, 1000.0, 20.0)
if kinetics in ['haldane', 'andrews']:
    params['K_I'] = st.sidebar.number_input("K_I [g/L]", 1.0, 1000.0, 250.0)
if kinetics == 'contois':
    params['K_C'] = st.sidebar.number_input("K_C [L/g]", 0.1, 50.0, 3.5)
if kinetics == 'teissier':
    params['K_T'] = st.sidebar.number_input("K_T [g/L]", 0.1, 50.0, 15.0)
if kinetics == 'linear':
    params['k'] = st.sidebar.number_input("k [1/day]", 0.001, 1.0, 0.05)
if kinetics == 'moser':
    params['n'] = st.sidebar.number_input("n (Moser exponent)", 1.0, 3.0, 1.5)
if kinetics == 'chen-hashimoto':
    params['S0'] = st.sidebar.number_input("S₀ [g/L]", 1.0, 1000.0, 100.0)
    params['k_CH'] = st.sidebar.number_input("k_CH", 0.01, 1.0, 0.2)
if kinetics == 'ierusalimsky':
    params['K_P'] = st.sidebar.number_input("K_P [g/L]", 1.0, 1000.0, 300.0)

st.sidebar.subheader("3️⃣ General Parameters")
params['Y_b'] = st.sidebar.number_input("Y_b (Biomass yield) [g B/g S]", 0.1, 1.0, 0.3)
params['Y_g'] = 1.0 - params['Y_b']
t1 = 0
S0 = st.sidebar.number_input("Initial substrate S₀ [g/L]", 0.1, 1000.0, 100.0)
B0 = st.sidebar.number_input("Initial biomass B₀ [g/L]", 0.01, 100.0, 1.0)
y0 = [S0, B0, 0.0]
t2 = st.sidebar.slider("Simulation time [days]", 10, 200, 50)

run = st.sidebar.button("Run Simulation")

# Model equations
def odefun(t, y, p):
    S_j, B, G = y
    mu_max = p.get('mu_max', 0.0)
    kinetics = p['kinetics']

    if kinetics == 'monod':
        mu = mu_max * S_j / (p['K_S'] + S_j)
        R_BS = mu * B
    elif kinetics == 'linear':
        R_BS = p['k'] * S_j
    elif kinetics == 'haldane':
        mu = mu_max * (S_j / (p['K_S'] + S_j)) * (p['K_I'] / (p['K_I'] + S_j))
        R_BS = mu * B
    elif kinetics == 'contois':
        mu = mu_max * ((S_j / B) / (p['K_C'] + S_j / B))
        R_BS = mu * B
    elif kinetics == 'teissier':
        mu = mu_max * (1 - np.exp(-S_j / p['K_T']))
        R_BS = mu * B
    elif kinetics == 'moser':
        mu = mu_max * S_j**p['n'] / (p['K_S'] + S_j**p['n'])
        R_BS = mu * B
    elif kinetics == 'chen-hashimoto':
        S_ratio = S_j / p['S0']
        mu = mu_max * S_ratio / (p['k_CH'] + S_ratio * (1 - S_ratio))
        R_BS = mu * B
    elif kinetics == 'andrews':
        mu = mu_max * S_j / (p['K_S'] + S_j + S_j**2 / p['K_I'])
        R_BS = mu * B
    elif kinetics == 'ierusalimsky':
        mu = mu_max * (S_j / (p['K_S'] + S_j)) * (p['K_P'] / (p['K_P'] + S_j))
        R_BS = mu * B
    else:
        raise ValueError(f"Unknown kinetics type: {kinetics}")

    R_sub = R_BS / p['Y_b']
    R_gas = p['Y_g'] * R_sub
    return [-R_sub, R_BS, R_gas]

# Show description or results
if not run:
    with st.expander(" Model Description", expanded=True):
        st.markdown(
            "This simulator implements the **Single-Step Degradation Model (SSDM)** for anaerobic digestion, "
            "based on [Fedailaine et al. 2015]. It assumes a **batch** reactor setup with **volume-normalized** "
            "concentrations of a degradable substrate $S_j$, microbial biomass $B$, and resulting cumulative biogas $G$."
        )

        st.markdown("---")
        st.markdown("####  Key Assumptions")
        st.markdown(
            "- The system is **closed**: no inflow or outflow (dilution rate $D = 0$)\n"
            "- Intermediate reactions are **ignored** (no metabolic stages)\n"
            "- Focus is on substrate uptake and biogas yield\n"
            "- You must specify the **kinetic model** and its parameters"
        )

        st.markdown("---")
        st.markdown("####  Mass Balances (Simplified)")
        st.latex(r"\frac{dS_j}{dt} = -\frac{R_{B/S_j}}{Y_b}")
        st.latex(r"\frac{dB}{dt} = R_{B/S_j}")
        st.latex(r"\frac{dG}{dt} = Y_g \cdot \frac{R_{B/S_j}}{Y_b}")

        st.markdown("Where:")
        st.markdown(
            "$S_j$ — degradable substrate concentration [g/L]  \n"
            "$B$ — microbial biomass [g/L]  \n"
            "$G$ — cumulative gas produced [g]  \n"
            "$R_{B/S_j}$ — rate of substrate conversion to biomass  \n"
            "$Y_b$ — biomass yield coefficient  \n"
            "$Y_g = 1 - Y_b$ — gas yield coefficient"
        )

        st.markdown("---")
        st.markdown(
            "This model helps estimate how much gas is produced from known feedstocks like **apple residues**, "
            "**straw**, or **manure**, over time under selected microbial dynamics."
        )



else:
    sol = solve_ivp(lambda t, y: odefun(t, y, params), [t1, t2], y0, t_eval=np.linspace(t1, t2, 300))
    t = sol.t
    S, B, G = sol.y
    D = S + B

    st.subheader(" Simulation Output")
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(t, S, label="Substrate", color="blue")
        ax.plot(t, B, label="Biomass", color="red")
        ax.plot(t, G, label="Biogas", color="green")
        ax.plot(t, D, label="Digestate", color="black", linestyle="--")
        ax.set_xlabel("Time [days]")
        ax.set_ylabel("Concentration [g/L]")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)


# Feedback Section
st.markdown("---")
st.subheader(" We value your feedback!")
st.markdown("Please feel free to leave some feedback below:")

with st.form("feedback_form"):
    name = st.text_input("Your Name")
    email = st.text_input("Your Email (optional)")
    message = st.text_area("Your Feedback", height=150)
    submitted = st.form_submit_button("Submit")

    if submitted:
        if not message.strip():
            st.warning("Feedback cannot be empty.")
        else:
            try:
                from email.mime.text import MIMEText
                import smtplib
                import os
                from dotenv import load_dotenv

                load_dotenv()
                EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
                EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
                TO_EMAIL = os.getenv("TO_EMAIL")

                content = f"Feedback from: {name}\nEmail: {email}\n\nMessage:\n{message}"
                msg = MIMEText(content)
                msg["Subject"] = "New Feedback from Streamlit App"
                msg["From"] = EMAIL_ADDRESS
                msg["To"] = TO_EMAIL

                with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                    server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                    server.send_message(msg)

                st.success("✅ Thank you! Your feedback has been sent.")
            except Exception as e:
                st.error(f"❌ An error occurred while sending feedback: {e}")

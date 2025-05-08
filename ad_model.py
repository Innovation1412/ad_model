import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import base64
from pathlib import Path
import streamlit.components.v1 as components



st.set_page_config(page_title="Anaerobic Digestion Model", layout="wide")
st.title("Anaerobic Digestion Simulator")

st.markdown("### 📘 Quick Start Guide")

#pdf_path = Path("problem1_description.pdf")
#if pdf_path.exists():
#    with open(pdf_path, "rb") as f:
#        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
#    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px" type="application/pdf"></iframe>'
#    st.markdown(pdf_display, unsafe_allow_html=True)
#else:
#    st.warning("📄 The PDF user guide (problem1_description.pdf) was not found in the app folder.")

pdf_path = Path("problem1_description.pdf")

if pdf_path.exists():
    with open(pdf_path, "rb") as f:
        pdf_data = f.read()
        base64_pdf = base64.b64encode(pdf_data).decode("utf-8")

    # Download button
    st.download_button(
        label="📄 Download PDF Guide",
        data=pdf_data,
        file_name="problem1_description.pdf",
        mime="application/pdf"
    )

else:
    st.warning("📄 The PDF user guide (problem1_description.pdf) was not found in the app folder.")


# Sidebar Controls
st.sidebar.subheader("1️⃣ Choose Kinetics")
kinetic_descriptions = {
    "monod": r"\mu = \mu_{\text{max}} \cdot \frac{S}{K_S + S}",
    "linear": r"\mu = k \cdot S",
    "haldane": r"\mu = \mu_{\text{max}} \cdot \frac{S}{K_S + S} \cdot \frac{K_I}{K_I + S}",
    "contois": r"\mu = \mu_{\text{max}} \cdot \frac{S/B}{K_C + S/B}",
    "teissier": r"\mu = \mu_{\text{max}} \cdot (1 - e^{-S/K_T})",
    "moser": r"\mu = \mu_{\text{max}} \cdot \frac{S^n}{K_S + S^n}",
    "chen-hashimoto": r"\mu = \mu_{\text{max}} \cdot \frac{S/S_0}{k_{CH} + \frac{S}{S_0}(1 - \frac{S}{S_0})}",
    "andrews": r"\mu = \mu_{\text{max}} \cdot \frac{S}{K_S + S + \frac{S^2}{K_I}}",
}
kinetics = st.sidebar.selectbox("Kinetic model", list(kinetic_descriptions.keys()))
st.sidebar.latex(kinetic_descriptions[kinetics])

params = {'kinetics': kinetics}

st.sidebar.subheader("2️⃣ Kinetic Parameters")
if kinetics in ['monod', 'haldane', 'contois', 'teissier', 'moser', 'chen-hashimoto', 'andrews']:
    params['mu_max'] = st.sidebar.number_input("μ_max [1/day]", 0.01, 5.0, 0.4)
if kinetics in ['monod', 'haldane', 'moser', 'andrews']:
    params['K_S'] = st.sidebar.number_input("K_S [g/L]", 0.1, 1000.0, 20.0, step=0.5)
if kinetics in ['haldane', 'andrews']:
    params['K_I'] = st.sidebar.number_input("K_I [g/L]", 1.0, 1000.0, 250.0, step=5.0)
if kinetics == 'contois':
    params['K_C'] = st.sidebar.number_input("K_C [L/g]", 0.1, 50.0, 3.5, step=0.1)
if kinetics == 'teissier':
    params['K_T'] = st.sidebar.number_input("K_T [g/L]", 0.1, 50.0, 15.0, step=0.1)
if kinetics == 'linear':
    params['k'] = st.sidebar.number_input("k [1/day]", 0.001, 1.0, 0.05)
if kinetics == 'moser':
    params['n'] = st.sidebar.number_input("n (Moser exponent)", 1.0, 3.0, 1.5)
if kinetics == 'chen-hashimoto':
    params['S0'] = st.sidebar.number_input("S₀ [g/L]", 1.0, 1000.0, 100.0, step = 2.0)
    params['k_CH'] = st.sidebar.number_input("k_CH", 0.01, 1.0, 0.2)

st.sidebar.subheader("3️⃣ General Parameters")
#params['Y_b'] = st.sidebar.number_input("Y_b (Biomass yield) [g B/g S]", 0.1, 1.0, 0.3)
params['Y_g'] = st.sidebar.number_input("Y_b (Biogas yield)", 0.0, 1.0, 0.314)
t1 = 0
S0 = st.sidebar.number_input("Initial substrate S₀ [g/L]", 0.1, 1000.0, 100.0, step = 2.0)
B0 = st.sidebar.number_input("Initial biomass B₀ [g/L]", 0.01, 100.0, 1.0)
y0 = [S0, B0, 0.0]
t2 = st.sidebar.slider("Simulation time [days]", 10, 200, 50, step = 1)

if 'simulate' not in st.session_state:
    st.session_state.simulate = False

if st.sidebar.button("Run Simulation"):
    st.session_state.simulate = True

# Model equations
    
def odefun(t, y, p):
    S_j, B, G = y
    kinetics = p['kinetics']

    if kinetics == 'monod':
        mu = p['mu_max'] * S_j / (p['K_S'] + S_j)
        R_BS_j = mu * B
    elif kinetics == 'linear':
        R_BS_j = p['k'] * S_j
    elif kinetics == 'haldane':
        mu = p['mu_max'] * (S_j / (p['K_S'] + S_j)) * (p['K_I'] / (p['K_I'] + S_j))
        R_BS_j = mu * B
    elif kinetics == 'contois':
        mu = p['mu_max'] * ((S_j / B) / (p['K_C'] + S_j / B))
        R_BS_j = mu * B
    elif kinetics == 'teissier':
        mu = p['mu_max'] * (1 - np.exp(-S_j / p['K_T']))
        R_BS_j = mu * B
    elif kinetics == 'moser':
        mu = p['mu_max'] * S_j ** p['n'] / (p['K_S'] + S_j ** p['n'])
        R_BS_j = mu * B
    elif kinetics == 'chen-hashimoto':
        S_ratio = S_j / p['S0']
        mu = p['mu_max'] * S_ratio / (p['k_CH'] + S_ratio * (1 - S_ratio))
        R_BS_j = mu * B
    elif kinetics == 'andrews':
        mu = p['mu_max'] * S_j / (p['K_S'] + S_j + S_j ** 2 / p['K_I'])
        R_BS_j = mu * B
    else:
        raise ValueError(f"Unknown kinetics type: {kinetics}")

    R_GS_j = p['Y_g'] * R_BS_j
    dS_j_dt = -R_BS_j - R_GS_j
    dB_dt = R_BS_j
    dG_dt = R_GS_j

    return [dS_j_dt, dB_dt, dG_dt]


# Show description or results
# with st.expander("Model Description", expanded=True):
    # st.markdown(
    # "This simulator implements the **Single-Step Degradation Model (SSDM)** for anaerobic digestion, "
    # "based on the work of Fedailaine et al., *Modeling of the anaerobic digestion of organic waste for biogas production*, "
    # "*Procedia Computer Science* (2015), **52**, 730–737. "
    # "[https://doi.org/10.1016/j.procs.2015.05.086](https://doi.org/10.1016/j.procs.2015.05.086)"
    # )

    # st.markdown(
        # "This model simulates the time-dependent degradation of a single substrate (input) "
        # "and its conversion to biogas (output) in a batch anaerobic digester. It predicts the biogas production rate "
        # "from a given feedstock mass at any point in time. "
        # "Intermediate steps and microbial interactions are ignored to focus on overall substrate-to-gas dynamics.\n\n"
        # "**When to use:**\n"
        # "1. You want to estimate how much gas is produced over time from a known feedstock.\n"
        # "2. Intermediate species and detailed pathways are not needed."
    # )

    # st.markdown("**The model is given as follows:**")
    # st.latex(r"\frac{d(S_j)}{dt} = D \cdot (S_j^{in} - S_j) + R_{S_j} - R_{B/S_j} - \frac{R_G}{S_j}")

    # st.markdown("**Term Definitions and Assumptions:**")
    # st.markdown(
        # "- **$D$** — Dilution rate (feed flow rate / reactor volume).  \n"
        # "- **$D \\cdot (S_j^{in} − S_j)$** — Represents the inflow and outflow of substrate $S_j$. If the system is batch-fed or semi-batch with no continuous inflow, set $D = 0$. This simplifies the model to describe degradation in a closed reactor.\n\n"
        # "- **$R_{S_j}$** — Net production of $S_j$ due to other reactions.  \n"
        # "  This is also set to zero, as such reactions are not expected here.  \n"
        # "  Total substrate mass is $S_j \\cdot V$ (concentration × volume).\n\n"
        # "- **$R_{B/S_j}$** — Rate of biochemical degradation or microbial uptake of substrate $S_j$. For example, glucose → biogas. This term removes $S_j$ from the system.  It may reflect conversion into intermediates or direct transformation into biogas.\n\n"
        # "- This rate depends on microbial kinetics, selectable via the sidebar:  \n"
        # "  **Monod**, **linear**, **Haldane**, **Contois**, **Teissier**, **Moser**, **Chen-Hashimoto**, **Andrews**, **Ierusalimsky**. For an overview of these models, see [Samuel Emebu et al., *Renewable and Sustainable Energy Reviews* (2023)](https://doi.org/10.1016/j.rser.2023.113671). \n"
        # "- **$\\mu_{max}$** — Maximum specific uptake rate.  \n"
        # "- **$K_S$** — Half-saturation constant (microbial affinity for substrate).  \n"
        # "- **$B$** — Biomass concentration (microbial mass per volume).\n\n"
        # "**Simplified model currently implemented:**  \n"
    # )
    # st.latex(r"\frac{dS_j}{dt} = - R_{B/S_j} - \frac{R_G}{S_j}")

    # st.markdown("---")
    # st.markdown("#### Key Assumptions")
    # st.markdown(
        # "- The system is **closed**: no inflow or outflow (dilution rate $D = 0$).  \n"
        # "- Intermediate reactions are **ignored** (no metabolic stages).  \n"
        # "- Focus is on substrate uptake and biogas yield.  \n"
        # "- You must specify the **kinetic model** and its parameters."
    # )

    # st.markdown("---")
    # st.markdown("#### Mass Balances (Simplified)")
    # st.latex(r"\frac{dS_j}{dt} = -\frac{R_{B/S_j}}{Y_b}")
    # st.latex(r"\frac{dB}{dt} = R_{B/S_j}")
    # st.latex(r"\frac{dG}{dt} = Y_g \cdot \frac{R_{B/S_j}}{Y_b}")

    # st.markdown("Where:")
    # st.markdown(
        # "$S_j$ — Degradable substrate concentration [g/L]  \n"
        # "$B$ — Microbial biomass [g/L]  \n"
        # "$G$ — Cumulative gas produced [g]  \n"
        # "$R_{B/S_j}$ — Rate of substrate conversion to biomass  \n"
        # "$Y_b$ — Biomass yield coefficient  \n"
        # "$Y_g = 1 - Y_b$ — Gas yield coefficient"
    # )

    # st.markdown("---")
    # st.markdown(
        # "This model helps estimate how much gas is produced from known feedstocks like **apple residues**, "
        # "**straw**, or **manure**, over time under selected microbial dynamics."
# )


if st.session_state.simulate:
    sol = solve_ivp(lambda t, y: odefun(t, y, params), [t1, t2], y0, t_eval=np.linspace(t1, t2, 300))
    t = sol.t
    S, B, G = sol.y

    st.subheader(" Simulation Output")
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(t, S, label="Substrate", color="blue")
        ax.plot(t, B, label="Biomass", color="red")
        ax.plot(t, G, label="Biogas", color="green")
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

"""
Aplicación Web en Streamlit para desplegar en la nube la simulación del
artículo de Quintanilla et al. (MethodsX, 2023).

Ejecución local:
    streamlit run app_web_streamlit.py
Despliegue en la nube (URL pública):
    Subir a GitHub y conectar en https://share.streamlit.io (Streamlit Community Cloud)
    o desplegar en VPS/Cloud con: streamlit run app_web_streamlit.py --server.port 8501
"""

import streamlit as st
import matplotlib.pyplot as plt
from simulacion_flotacion_pid import (
    SIMC_PARAMS_FINAL,
    SIMC_PARAMS_INITIAL,
    THETA_S,
    simular_banco_flotacion,
)

st.set_page_config(
    page_title="Simulación Control PID - Banco de Flotación",
    layout="wide",
)

st.title("Control de Nivel Feedback-Feedforward en Banco de Flotación (3 Celdas)")
st.caption(
    "Réplica del modelo matemático integrador con retardo y sintonización SIMC "
    "(Quintanilla et al., MethodsX 2023)."
)

with st.sidebar:
    st.header("Parámetros de Simulación")
    duracion_s = st.slider("Duración [s]", 300, 900, 600, step=60)
    preset = st.selectbox(
        "Sintonización SIMC (Tabla 2)",
        ["Final (Tabla 2)", "Inicial SIMC (Tabla 2)", "Personalizado"],
    )
    ruido_std = st.slider("Ruido sensor por aireación [cm]", 0.0, 0.4, 0.12, step=0.02)
    usar_filtro = st.checkbox("Filtro de mediana móvil (10 s)", value=True)

    base_params = (
        SIMC_PARAMS_INITIAL if preset == "Inicial SIMC (Tabla 2)" else SIMC_PARAMS_FINAL
    )
    custom_params = []
    for i in range(3):
        with st.expander(f"Celda {i+1} (θ = {THETA_S[i]} s)", expanded=(i == 0)):
            kc = st.number_input(
                f"Kc_{i+1} [lpm/cm]",
                value=float(base_params[i]["Kc"]),
                step=1.0,
                disabled=(preset != "Personalizado"),
            )
            tau_i = st.number_input(
                f"τ_i{i+1} [s]",
                value=float(base_params[i]["tau_i"]),
                min_value=1.0,
                step=1.0,
                disabled=(preset != "Personalizado"),
            )
            tau_d = st.number_input(
                f"τ_d{i+1} [s]",
                value=float(base_params[i]["tau_d"]),
                min_value=0.0,
                step=0.5,
                disabled=(preset != "Personalizado"),
            )
            k_ff = st.slider(
                f"K_ff{i+1}",
                0.0,
                1.5,
                float(base_params[i]["K_ff"]),
                step=0.05,
                disabled=(preset != "Personalizado"),
            )
            custom_params.append(
                {"Kc": kc, "tau_i": tau_i, "tau_d": tau_d, "K_ff": k_ff}
            )

res_fb = simular_banco_flotacion(
    duracion_s=duracion_s,
    modo="feedback",
    params_celdas=custom_params,
    ruido_std_cm=ruido_std,
    usar_filtro_mediana=usar_filtro,
)
res_ff = simular_banco_flotacion(
    duracion_s=duracion_s,
    modo="feedback_feedforward",
    params_celdas=custom_params,
    ruido_std_cm=ruido_std,
    usar_filtro_mediana=usar_filtro,
)

cols = st.columns(3)
for i, col in enumerate(cols):
    mejora = (res_fb["iae"][i] - res_ff["iae"][i]) / max(res_fb["iae"][i], 1e-6) * 100.0
    col.metric(
        label=f"Celda {i+1} — IAE Feedback+FF",
        value=f"{res_ff['iae'][i]:.1f} cm·s",
        delta=f"{-mejora:.1f}% vs Feedback ({res_fb['iae'][i]:.1f})",
        delta_color="inverse",
    )

fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
for i in range(3):
    for col_idx, (res, nombre) in enumerate(
        [(res_fb, "Feedback-only"), (res_ff, "Feedback + Feedforward")]
    ):
        ax = axes[i, col_idx]
        ax.plot(res["t"], res["setpoints"][i], "r-", lw=1.5, label="Setpoint")
        ax.plot(res["t"], res["h_filt"][i], "b-", lw=1.4, label="Nivel h_p")
        ax.set_ylim(24, 38)
        ax.set_ylabel("h_p [cm]", color="b")
        ax.grid(True, alpha=0.25)

        ax2 = ax.twinx()
        ax2.plot(res["t"], res["u_cmd"][i], "k-", lw=1.0, alpha=0.8, label="Q_tailings")
        ax2.plot(res["t"], res["q_dist"][i], "c--", lw=1.0, alpha=0.8, label="Q_d")
        ax2.set_ylim(0, 105)
        ax2.set_ylabel("Caudal [lpm]")
        ax.set_title(f"{nombre} - Celda {i+1} (IAE = {res['iae'][i]:.1f} cm·s)")

axes[2, 0].set_xlabel("Tiempo [s]")
axes[2, 1].set_xlabel("Tiempo [s]")
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

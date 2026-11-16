"""
Simulación del Control de Nivel Feedback y Feedback-Feedforward (PI/PID + SIMC)
en un Banco de Flotación de 3 Celdas en Serie con Bombas Peristálticas.

Basado íntegramente en el artículo:
Quintanilla, P., Navia, D., Moreno, F., Neethling, S. J., & Brito-Parada, P. R. (2023).
"A methodology to implement a closed-loop feedback-feedforward level control in a
laboratory-scale flotation bank using peristaltic pumps."
MethodsX, 10, 102081. https://doi.org/10.1016/j.mex.2023.102081
"""

import numpy as np
import matplotlib.pyplot as plt


# ==============================================================================
# 1. PARÁMETROS DEL SISTEMA EXPERIMENTAL Y MODELO MATEMÁTICO (Tabla 1, Tabla 2, Figs. 8-12)
# ==============================================================================

# Dimensiones geométricas de cada celda y tanque de alimentación T-1 (Pág. 2, 6; Fig. 2)
ANCHO_CELDA_CM = 30.0
LARGO_CELDA_CM = 30.0
ALTO_CELDA_CM = 45.0
AREA_CELDA_CM2 = ANCHO_CELDA_CM * LARGO_CELDA_CM  # 900 cm^2
CAPACIDAD_TANQUE_L = 130.0  # Tanque de alimentación T-1 en lazo cerrado (130 L)
SENSOR_NIVEL_MIN_CM = 0.0   # Límite inferior rango calibrado sensor óptico (Pág. 6)
SENSOR_NIVEL_MAX_CM = 39.0  # Límite superior rango calibrado sensor óptico (Pág. 6)

# Flujo de aire constante en cada celda (pág. 7 y 13)
Q_AIR_LPM = 47.0
J_G_CM_S = Q_AIR_LPM * (1000.0 / 60.0) / AREA_CELDA_CM2  # ~0.87 cm/s

# Pendiente integradora experimental k' = dy / (dt * du) [cm / (s * lpm)] (Tabla 2, Ec. 5)
K_PRIME = -0.001018

# Retardo de tiempo (dead time) theta_i [s] de cada celda (Tabla 2)
THETA_S = [7, 6, 3]

# Parámetros de calibración PLC 16-bit (0 - 32,767 bits) <-> Unidades de ingeniería
# Bombas peristálticas: Q [lpm] = m_pump * bits (Fig. 9)
PUMP_SLOPES = {
    "feed": 0.0035,   # P-4: Q_feed = 0.0035 * x
    "cell1": 0.0029,  # P-1: Q_tailings1 = 0.0029 * x
    "cell2": 0.0033,  # P-2: Q_tailings2 = 0.0033 * x
    "cell3": 0.0033,  # P-3: Q_tailings3 = 0.0033 * x
}
PLC_MAX_BITS = 32767

# Sensores ópticos de nivel: h_p [cm] = m_lvl * bits + b_lvl (Fig. 11)
LEVEL_SENSOR_CAL = [
    {"m": -0.0018, "b": 40.1363},  # Celda 1
    {"m": -0.0019, "b": 43.8823},  # Celda 2
    {"m": -0.0020, "b": 41.2930},  # Celda 3
]

# Parámetros de sintonización SIMC (Tabla 2)
SIMC_PARAMS_FINAL = [
    {"Kc": -62.0, "tau_i": 56.0, "tau_d": 0.0, "K_ff": 0.8},  # Celda 1
    {"Kc": -76.0, "tau_i": 30.0, "tau_d": 0.0, "K_ff": 0.8},  # Celda 2
    {"Kc": -85.0, "tau_i": 44.0, "tau_d": 0.0, "K_ff": 1.0},  # Celda 3
]

SIMC_PARAMS_INITIAL = [
    {"Kc": -70.2, "tau_i": 56.0, "tau_d": 0.0, "K_ff": 0.8},  # Celda 1 (tau_c=7)
    {"Kc": -81.8, "tau_i": 48.0, "tau_d": 0.0, "K_ff": 0.8},  # Celda 2 (tau_c=6)
    {"Kc": -78.1, "tau_i": 52.0, "tau_d": 0.0, "K_ff": 1.0},  # Celda 3 (tau_c=8)
]


def cm_to_plc_bits(h_cm: float, cell_idx: int) -> int:
    """Convierte nivel en cm (rango calibrado 0-39 cm) a entero 16-bit PLC (0-32767)."""
    cal = LEVEL_SENSOR_CAL[cell_idx]
    h_clamped = float(np.clip(h_cm, SENSOR_NIVEL_MIN_CM, SENSOR_NIVEL_MAX_CM))
    bits = (h_clamped - cal["b"]) / cal["m"]
    return int(np.clip(round(bits), 0, PLC_MAX_BITS))


def plc_bits_to_cm(bits: int, cell_idx: int) -> float:
    """Convierte registro entero de 16 bits del PLC a nivel en cm (Fig. 11)."""
    cal = LEVEL_SENSOR_CAL[cell_idx]
    return cal["m"] * bits + cal["b"]


def lpm_to_plc_bits(q_lpm: float, pump_key: str) -> int:
    """Convierte caudal en lpm a señal de salida del PLC en bits (0-32767)."""
    bits = q_lpm / PUMP_SLOPES[pump_key]
    return int(np.clip(round(bits), 0, PLC_MAX_BITS))


def plc_bits_to_lpm(bits: int, pump_key: str) -> float:
    """Convierte señal de salida del PLC en bits a caudal en lpm (Fig. 9)."""
    return PUMP_SLOPES[pump_key] * bits


# ==============================================================================
# 2. MOTOR DE SIMULACIÓN DEL BANCO DE FLOTACIÓN (Ecs. 1, 4, 7, 8, 9, 10)
# ==============================================================================

def generar_perfiles_referencia(n_steps: int, ts: float = 1.0):
    """
    Genera los perfiles de setpoint (h_sp) y caudal de alimentación (Q_feed)
    compatibles con los experimentos de validación de la Fig. 14.
    """
    t = np.arange(n_steps) * ts

    # Setpoints escalonados por celda [cm] (rango operativo 0-39 cm)
    sp1 = np.where(t < 180, 30.0, np.where(t < 420, 29.0, 29.5))
    sp2 = np.where(t < 150, 34.0, np.where(t < 380, 33.0, 33.5))
    sp3 = np.where(t < 200, 33.0, np.where(t < 450, 32.0, 32.5))

    # Perturbación de alimentación Q_feed [lpm] (45 lpm nominal con variación opcional)
    q_feed = np.full(n_steps, 45.0)
    # Escalón de perturbación suave para evidenciar rechazo de perturbaciones
    q_feed[(t >= 260) & (t < 320)] = 52.0
    q_feed[(t >= 480) & (t < 540)] = 40.0

    return t, [sp1, sp2, sp3], q_feed


def simular_banco_flotacion(
    duracion_s: int = 600,
    ts: float = 1.0,
    modo: str = "feedback_feedforward",
    params_celdas=None,
    ruido_std_cm: float = 0.12,
    usar_filtro_mediana: bool = True,
    seed: int = 42,
):
    """
    Simula las 3 celdas de flotación conectadas en serie (Fig. 15) con:
      - Modelo integrador con retardo: dh_i/dt = k' * (u_i(t - theta_i) - Q_di(t))
      - Conversión PLC 16-bit y filtro de mediana móvil de 10 s (medfilt1)
      - Controlador PI/PID estándar (Ec. 1) + Feedforward (Ecs. 7 y 8)
      - Saturación física de bombas peristálticas y anti-windup
      - Cálculo de IAE (Ec. 9)

    Parámetros:
        duracion_s: Duración total de la simulación en segundos (600 s o 900 s).
        ts: Tiempo de muestreo [s] (1.0 s en el artículo).
        modo: "feedback" (solo realimentación) o "feedback_feedforward".
        params_celdas: Lista de dicts con Kc, tau_i, tau_d, K_ff para cada celda.
        ruido_std_cm: Desviación estándar del ruido por turbulencia de aire [cm].
        usar_filtro_mediana: Aplica filtro de mediana móvil de ventana 10 s.
        seed: Semilla aleatoria para reproducibilidad del ruido.
    """
    if params_celdas is None:
        params_celdas = SIMC_PARAMS_FINAL

    rng = np.random.default_rng(seed)
    n_steps = int(duracion_s / ts)
    t, setpoints, q_feed = generar_perfiles_referencia(n_steps, ts)

    pump_keys = ["cell1", "cell2", "cell3"]
    u_max = [PUMP_SLOPES[k] * PLC_MAX_BITS for k in pump_keys]
    u0 = float(q_feed[0])

    # Estados iniciales en estado estacionario (nivel en setpoint inicial, caudal = Q_feed[0])
    h_true = np.zeros((3, n_steps))
    h_meas_raw = np.zeros((3, n_steps))
    h_filt = np.zeros((3, n_steps))
    u_cmd = np.zeros((3, n_steps))
    q_dist = np.zeros((3, n_steps))

    for i in range(3):
        h_true[i, 0] = setpoints[i][0]
        bits_init = cm_to_plc_bits(h_true[i, 0], i)
        h_meas_raw[i, 0] = plc_bits_to_cm(bits_init, i)
        h_filt[i, 0] = h_meas_raw[i, 0]
        u_cmd[i, 0] = u0
        q_dist[i, 0] = u0

    integral_err = [0.0, 0.0, 0.0]
    prev_err = [0.0, 0.0, 0.0]

    # Bucle de simulación en tiempo discreto (Ts = 1 s)
    for k in range(n_steps):
        for i in range(3):
            # 1. Perturbación de entrada a la celda i (Fig. 15)
            # Celda 1 recibe Q_feed; Celda 2 recibe Q_tailings1; Celda 3 recibe Q_tailings2
            if i == 0:
                q_in = q_feed[k]
                q_in_prev = q_feed[k - 1] if k > 0 else q_feed[0]
            else:
                q_in = u_cmd[i - 1, k]
                q_in_prev = u_cmd[i - 1, k - 1] if k > 0 else u_cmd[i - 1, 0]
            q_dist[i, k] = q_in

            # 2. Lectura del sensor de nivel con ruido por inyección de aire + cuantización PLC 16-bit
            if k > 0:
                ruido = rng.normal(0.0, ruido_std_cm)
                bits_sensor = cm_to_plc_bits(h_true[i, k] + ruido, i)
                h_meas_raw[i, k] = plc_bits_to_cm(bits_sensor, i)

            # 3. Filtro de mediana móvil de 10 s (medfilt1 causal con ventana de 10 muestras)
            if usar_filtro_mediana:
                win_start = max(0, k - 9)
                h_filt[i, k] = float(np.median(h_meas_raw[i, win_start : k + 1]))
            else:
                h_filt[i, k] = h_meas_raw[i, k]

            # 4. Cálculo del error: e(t) = y_sp(t) - y(t)
            err = setpoints[i][k] - h_filt[i, k]
            p = params_celdas[i]
            kc = p["Kc"]
            tau_i = p["tau_i"]
            tau_d = p.get("tau_d", 0.0)
            k_ff = p["K_ff"] if modo == "feedback_feedforward" else 0.0

            # Término derivativo (0 en modo PI del artículo)
            d_err = (err - prev_err[i]) / ts if k > 0 else 0.0

            # Término feedforward: K_ff * (Q_di(t) - Q_di(t-1)) (Ecs. 7 y 8)
            delta_q = q_in - q_in_prev
            u_ff = k_ff * delta_q

            # Señal de control PI/PID (alrededor del punto de operación nominal u0 = Q_feed[0])
            u_fb_tentative = u0 + kc * (
                err + (integral_err[i] + err * ts) / tau_i + tau_d * d_err
            )
            u_total = u_fb_tentative + u_ff

            # Saturación física de la bomba peristáltica [0, u_max] y anti-windup condicional
            u_sat = float(np.clip(u_total, 0.0, u_max[i]))
            saturated_high = u_total >= u_max[i] and (kc * err / tau_i) > 0.0
            saturated_low = u_total <= 0.0 and (kc * err / tau_i) < 0.0
            if not (saturated_high or saturated_low):
                integral_err[i] += err * ts

            # Cuantización a 16-bit PLC de salida (AQ00X)
            bits_act = lpm_to_plc_bits(u_sat, pump_keys[i])
            u_cmd[i, k] = plc_bits_to_lpm(bits_act, pump_keys[i])
            prev_err[i] = err

        # 5. Integración del modelo dinámico de la planta para el siguiente paso k+1
        # dh_i/dt = k' * (u_i(t - theta_i) - Q_di(t))
        if k < n_steps - 1:
            for i in range(3):
                delay_steps = int(round(THETA_S[i] / ts))
                idx_delayed = max(0, k - delay_steps)
                u_delayed = u_cmd[i, idx_delayed]
                dh_dt = K_PRIME * (u_delayed - q_dist[i, k])
                h_next = h_true[i, k] + dh_dt * ts
                h_true[i, k + 1] = float(np.clip(h_next, 0.0, ALTO_CELDA_CM))

    # Cálculo del Error Absoluto Integral (IAE) por celda (Ec. 9)
    iae = [float(np.sum(np.abs(setpoints[i] - h_filt[i])) * ts) for i in range(3)]

    return {
        "t": t,
        "setpoints": setpoints,
        "h_true": h_true,
        "h_filt": h_filt,
        "u_cmd": u_cmd,
        "q_dist": q_dist,
        "iae": iae,
    }


# ==============================================================================
# 3. VISUALIZACIÓN COMPARATIVA (Replicación de la Figura 14)
# ==============================================================================

def graficar_comparacion(res_fb, res_ff, guardar_ruta: str = None):
    """Genera las gráficas comparativas Feedback-only vs Feedback+Feedforward."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    fig.suptitle(
        "Control de Nivel en Banco de Flotación (3 Celdas en Serie) - Modelo SIMC PI + Feedforward",
        fontsize=13,
        fontweight="bold",
    )

    for i in range(3):
        for col, (res, titulo) in enumerate(
            [(res_fb, "Feedback-only"), (res_ff, "Feedback + Feedforward")]
        ):
            ax = axes[i, col]
            t = res["t"]

            # Eje izquierdo superior visualizado: Nivel h_p [cm]
            l1, = ax.plot(t, res["setpoints"][i], "r-", lw=1.6, label="Setpoint")
            l2, = ax.plot(t, res["h_filt"][i], "b-", lw=1.4, label="Nivel filtrado")
            ax.set_ylabel("h_p [cm]", color="b", fontsize=9)
            ax.set_ylim(24, 38)
            ax.grid(True, alpha=0.25)

            # Eje derecho: Caudales u(t) y perturbación Q_d(t) [lpm]
            ax2 = ax.twinx()
            l3, = ax2.plot(
                t, res["u_cmd"][i], "k-", lw=1.1, alpha=0.85, label="Q_tailings (u)"
            )
            l4, = ax2.plot(
                t, res["q_dist"][i], "c--", lw=1.1, alpha=0.85, label="Perturbación (Q_d)"
            )
            ax2.set_ylabel("Caudal [lpm]", color="k", fontsize=9)
            ax2.set_ylim(0, 105)

            ax.set_title(
                f"{titulo} - Celda {i+1} | IAE = {res['iae'][i]:.1f} cm·s",
                fontsize=10,
                fontweight="bold",
            )

            if i == 0 and col == 1:
                lines = [l1, l2, l3, l4]
                labels = [ln.get_label() for ln in lines]
                ax.legend(lines, labels, loc="upper right", fontsize=8, framealpha=0.9)

    axes[2, 0].set_xlabel("Tiempo [s]")
    axes[2, 1].set_xlabel("Tiempo [s]")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if guardar_ruta:
        plt.savefig(guardar_ruta, dpi=150)
    plt.show()


if __name__ == "__main__":
    print("=== Simulación Banco de Flotación (Quintanilla et al., MethodsX 2023) ===")
    res_fb = simular_banco_flotacion(duracion_s=600, modo="feedback")
    res_ff = simular_banco_flotacion(duracion_s=600, modo="feedback_feedforward")

    for i in range(3):
        mejora = (res_fb["iae"][i] - res_ff["iae"][i]) / res_fb["iae"][i] * 100.0
        print(
            f"Celda {i+1}: IAE Feedback = {res_fb['iae'][i]:.2f} cm·s | "
            f"IAE Feedback+FF = {res_ff['iae'][i]:.2f} cm·s | "
            f"Mejora = {mejora:.1f}%"
        )

    graficar_comparacion(res_fb, res_ff)

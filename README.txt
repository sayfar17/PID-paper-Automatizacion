================================================================================
SIMULACIÓN DE CONTROL DE NIVEL PI/PID + FEEDFORWARD EN BANCO DE FLOTACIÓN
================================================================================

Basado íntegramente en el artículo científico:
  Quintanilla, P., Navia, D., Moreno, F., Neethling, S. J., & Brito-Parada, P. R. (2023).
  "A methodology to implement a closed-loop feedback-feedforward level control in a
   laboratory-scale flotation bank using peristaltic pumps."
  MethodsX, 10, 102081. https://doi.org/10.1016/j.mex.2023.102081

--------------------------------------------------------------------------------
1. ESTRUCTURA DE ARCHIVOS DEL PROYECTO
--------------------------------------------------------------------------------
- simulacion_flotacion_pid.py   : Motor matemático completo en Python, calibración
                                  PLC 16-bit, controlador PI/PID + Feedforward y
                                  gráficas comparativas tipo Figura 14.
- app_web_streamlit.py          : Interfaz web interactiva en Streamlit (importa el
                                  motor de simulacion_flotacion_pid.py).
- simulacion_flotacion_pid.html : Réplica exacta 100% en navegador (HTML/JS), útil
                                  para ejecutar sin instalar Python.
- requirements.txt              : Dependencias de Python (numpy, scipy, matplotlib,
                                  streamlit).
- render.yaml / .streamlit/     : Configuración lista para despliegue en Render.

--------------------------------------------------------------------------------
2. RELACIÓN DEL CÓDIGO CON EL ARTÍCULO (MAPA ECUACIONES / TABLAS / FIGURAS)
--------------------------------------------------------------------------------
a) Geometría y condiciones de operación (Págs. 2, 6, 7, 13; Fig. 2):
   - 3 celdas rectangulares en serie de 30 x 30 cm (900 cm²) y 45 cm de altura.
   - Tanque de alimentación T-1 en lazo cerrado: 130 L.
   - Rango calibrado de los sensores ópticos de nivel: 0–39 cm.
   - Flujo de aire constante: Q_air = 47 lpm (j_g ≈ 0.87 cm/s).

b) Modelo matemático dinámico e integración (Ecs. 4, 5, 10; Tabla 2; Fig. 12):
   - Función de transferencia integradora con retardo por celda:
       G_p(s) = k' * exp(-theta_i * s) / s
   - Pendiente experimental: k' = -0.001018 cm·s⁻¹·lpm⁻¹ en las 3 celdas.
   - Retardos (dead time): theta = [7, 6, 3] s para Celda 1, Celda 2 y Celda 3.
   - En tiempo discreto (Ts = 1 s): dh_i/dt = k' * (u_i(t - theta_i) - Q_di(t)).

c) Acoplamiento hidráulico de perturbaciones en serie (Pág. 14; Fig. 15):
   - Celda 1: Q_d1 = Q_feed (45 lpm nominal con escalones de prueba 52 y 40 lpm).
   - Celda 2: Q_d2 = Q_tailings1 (caudal de salida de la bomba P-1).
   - Celda 3: Q_d3 = Q_tailings2 (caudal de salida de la bomba P-2).

d) Calibración PLC GE Fanuc 16-bit [0 – 32,767 bits] (Págs. 6–9; Figs. 9 y 11):
   - Bombas peristálticas (Q = m * bits):
       P-4 (Feed):    0.0035 * x
       P-1 (Celda 1): 0.0029 * x
       P-2 (Celda 2): 0.0033 * x
       P-3 (Celda 3): 0.0033 * x
   - Sensores ópticos de nivel (h_p = m * bits + b):
       Celda 1: -0.0018 * x + 40.1363
       Celda 2: -0.0019 * x + 43.8823
       Celda 3: -0.0020 * x + 41.2930
   - Funciones cm_to_plc_bits, plc_bits_to_cm, lpm_to_plc_bits y plc_bits_to_lpm
     aplican redondeo y saturación estricta a [0, 32767] bits.

e) Controlador PI/PID, sintonización SIMC y Feedforward (Ecs. 1, 7, 8; Tabla 2):
   - Ley de control implementada alrededor del punto nominal u0 = 45 lpm:
       u_i(t) = u0 + Kc_i * (e_i(t) + (1/tau_ii)*∫e_i(τ)dτ + tau_di*de_i/dt)
                + K_ffi * ΔQ_di(t)
     donde e_i(t) = y_sp,i(t) - y_i(t) y ΔQ_di(t) = Q_di(t) - Q_di(t - 1).
   - tau_d = 0 por defecto (PI puro como en el artículo), configurable a > 0 (PID).
   - Parámetros Iniciales SIMC (Tabla 2):
       Kc = [-70.2, -81.8, -78.1] lpm/cm, tau_i = [56, 48, 52] s.
   - Parámetros Finales (Tabla 2):
       Kc = [-62, -76, -85] lpm/cm, tau_i = [56, 30, 44] s, K_ff = [0.8, 0.8, 1.0].
   - Incluye saturación física de bombas [0, u_max] (Pág. 14) y anti-windup
     condicional para evitar acumulación del término integral bajo saturación.

f) Filtrado, muestreo y métrica de desempeño (Pág. 13; Ec. 9; Fig. 14):
   - Tiempo de muestreo Ts = 1 s.
   - Filtro de mediana móvil causal de 10 s sobre las lecturas de nivel (medfilt1).
   - Error Absoluto Integral: IAE_i = ∫ |e_i(t)| dt [cm·s] para comparar
     Feedback-only vs. Feedback + Feedforward.

--------------------------------------------------------------------------------
3. GUÍA DE DESPLIEGUE LOCAL PARA EL DOCENTE
--------------------------------------------------------------------------------
Requisitos previos: Python 3.10 o superior instalado (o usar Opción C sin Python).

--- OPCIÓN A: Ejecución directa del script Python (consola + gráficos Matplotlib) ---
1. Abrir una terminal (PowerShell / CMD / Terminal) dentro de esta carpeta.
2. (Opcional recomendado) Crear y activar un entorno virtual:
     python -m venv .venv
     .venv\Scripts\activate        (en Windows)
     source .venv/bin/activate     (en Linux/Mac)
3. Instalar dependencias:
     pip install -r requirements.txt
4. Ejecutar la simulación:
     python simulacion_flotacion_pid.py
   -> Imprime en consola los valores de IAE Feedback vs. Feedback+FF y muestra
      la figura comparativa equivalente a la Fig. 14 del artículo.

--- OPCIÓN B: Interfaz web interactiva local con Streamlit ---
1. Con las dependencias instaladas (paso 3 anterior), ejecutar:
     streamlit run app_web_streamlit.py
   (Si el comando "streamlit" no es reconocido en Windows, usar:
     python -m streamlit run app_web_streamlit.py)
2. Se abrirá automáticamente en el navegador en:
     http://localhost:8501
3. Desde la barra lateral el docente podrá:
   - Cambiar la duración (300–900 s).
   - Alternar entre "Final (Tabla 2)", "Inicial SIMC (Tabla 2)" o "Personalizado"
     (para modificar Kc, tau_i, tau_d > 0 en modo PID y K_ff por celda).
   - Ajustar el ruido por aireación y activar/desactivar el filtro de mediana 10 s.

--- OPCIÓN C: Ejecución inmediata sin instalar Python (solo navegador) ---
1. Hacer doble clic sobre el archivo "simulacion_flotacion_pid.html".
2. Se abrirá en Chrome/Edge/Firefox con los mismos parámetros, ecuaciones y
   controles interactivos de forma 100% offline.
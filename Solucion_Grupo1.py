import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cargar el modelo previamente entrenado
model = joblib.load("modelo_xgboost_hiperparametrizado.pkl")

# Título de la app
st.title("🔍 Predicción de Cancelación de Reserva de Hotel")

st.write("Completa la información de la reserva para predecir si será cancelada o no.")

# ======================
# Formulario de entrada
# ======================

hotel = st.selectbox("Tipo de hotel", ['City Hotel', 'Resort Hotel'])
tiempo_espera = st.number_input("Días de espera antes del check-in", min_value=0)
mes_llegada = st.selectbox("Mes de llegada", list(range(1, 13)))
finde = st.number_input("Noches de fin de semana", min_value=0)
semana = st.number_input("Noches entre semana", min_value=0)
adultos = st.number_input("Número de adultos", min_value=0)
ninos = st.number_input("Número de niños", min_value=0)
comida = st.selectbox("Tipo de comida", ['BB', 'HB', 'FB', 'SC'])
segmento = st.selectbox("Segmento de mercado", ['Online TA', 'Offline TA/TO', 'Direct', 'Groups'])
repetido = st.selectbox("¿Es huésped repetido?", ['Sí', 'No'])
habitacion = st.selectbox("Tipo de habitación reservada", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
cambios = st.number_input("Número de cambios en la reserva", min_value=0)
deposito = st.selectbox("Tipo de depósito", ['No Deposit', 'Refundable', 'Non Refund'])
tarifa = st.number_input("Tarifa diaria promedio (ADR)", min_value=0.0)
solicitudes = st.number_input("Total de solicitudes especiales", min_value=0)

# =========================
# Codificación de variables
# =========================

repetido_bin = 1 if repetido == 'Sí' else 0
hotel_bin = 1 if hotel == 'Resort Hotel' else 0
comida_map = {'BB':0, 'HB':1, 'FB':2, 'SC':3}
segmento_map = {'Online TA':0, 'Offline TA/TO':1, 'Direct':2, 'Groups':3}
habitacion_map = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6}
deposito_map = {'No Deposit':0, 'Refundable':1, 'Non Refund':2}

# Crear DataFrame con los valores de entrada
input_data = pd.DataFrame([[
    hotel_bin, tiempo_espera, mes_llegada, finde, semana, adultos, ninos,
    comida_map[comida], segmento_map[segmento], repetido_bin,
    habitacion_map[habitacion], cambios, deposito_map[deposito],
    tarifa, solicitudes
]], columns=[
    'hotel', 'tiempo_espera', 'mes_fecha_llegada', 'estancias_noches_fin_semana',
    'estancias_noches_semana', 'adultos', 'ninos', 'comida', 'segmento_mercado',
    'es_huesped_repetido', 'tipo_habitacion_reservada', 'cambios_reserva',
    'tipo_deposito', 'tarifa_diaria_promedio', 'total_solicitudes_especiales'
])

# ===================
# Predicción del modelo
# ===================

if st.button("Predecir Cancelación"):
    resultado = model.predict(input_data)[0]
    if resultado == 1:
        st.error("⚠️ La reserva probablemente será cancelada.")
    else:
        st.success("✅ La reserva probablemente no será cancelada.")
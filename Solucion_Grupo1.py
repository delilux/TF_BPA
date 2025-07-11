import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cargar modelo entrenado
model = joblib.load("modelo_xgboost_hiperparametrizado.pkl")

# Lista completa de columnas que espera el modelo (usa solo una muestra si son muchas)
columnas = [
    'hotel_Resort Hotel', 'comida_FB', 'comida_HB', 'comida_SC', 'comida_Undefined',
    'pais_PRT', 'pais_ESP', 'pais_FRA', 'pais_BRA', 'pais_GBR', 'pais_USA',
    'segmento_mercado_Online TA', 'segmento_mercado_Direct',
    'canal_distribucion_TA/TO', 'canal_distribucion_Direct',
    'tipo_habitacion_reservada_A', 'tipo_habitacion_reservada_B',
    'tipo_habitacion_asignada_A', 'tipo_habitacion_asignada_B',
    'tipo_deposito_Non Refund', 'tipo_deposito_Refundable',
    'tipo_cliente_Transient', 'tipo_cliente_Transient-Party',
    'mes_fecha_llegada', 'tiempo_espera', 'ano_fecha_llegada',
    'numero_semana_fecha_llegada', 'dia_del_mes_fecha_llegada',
    'estancias_noches_fin_semana', 'estancias_noches_semana', 'adultos',
    'ninos', 'bebes', 'es_huesped_repetido', 'cancelaciones_previas',
    'reservas_anteriores_no_canceladas', 'cambios_reserva', 'dias_en_lista_espera',
    'tarifa_diaria_promedio', 'plazas_aparcamiento_solicitadas', 'total_solicitudes_especiales'
]

# Crear un DataFrame base en blanco con todas las columnas en 0
input_data = pd.DataFrame(np.zeros((1, len(columnas))), columns=columnas)

st.title("🧾 Predicción de Cancelación de Reserva")

# Formulario de entrada
hotel = st.selectbox("Tipo de hotel", ['City Hotel', 'Resort Hotel'])
comida = st.selectbox("Tipo de comida", ['BB', 'HB', 'FB', 'SC', 'Undefined'])
pais = st.selectbox("País de origen", ['PRT', 'ESP', 'FRA', 'BRA', 'GBR', 'USA'])
segmento = st.selectbox("Segmento de mercado", ['Online TA', 'Direct'])
canal = st.selectbox("Canal de distribución", ['TA/TO', 'Direct'])
habitacion_reservada = st.selectbox("Habitación reservada", ['A', 'B'])
habitacion_asignada = st.selectbox("Habitación asignada", ['A', 'B'])
deposito = st.selectbox("Tipo de depósito", ['No Deposit', 'Refundable', 'Non Refund'])
tipo_cliente = st.selectbox("Tipo de cliente", ['Transient', 'Transient-Party'])
# Valores numéricos
mes = st.slider("Mes de llegada", 1, 12)
ano = st.selectbox("Año de llegada", [2016, 2017])
semana = st.slider("Semana del año", 1, 52)
dia = st.slider("Día del mes", 1, 31)
tiempo_espera = st.number_input("Tiempo de espera (días)", min_value=0)
noches_semana = st.number_input("Noches entre semana", min_value=0)
noches_finde = st.number_input("Noches de fin de semana", min_value=0)
adultos = st.number_input("Cantidad de adultos", min_value=0)
ninos = st.number_input("Cantidad de niños", min_value=0)
bebes = st.number_input("Cantidad de bebés", min_value=0)
repetido = st.selectbox("¿Huésped repetido?", ['No', 'Sí'])
cancelaciones_previas = st.number_input("Cancelaciones previas", min_value=0)
reservas_previas_ok = st.number_input("Reservas previas no canceladas", min_value=0)
cambios = st.number_input("Cambios en la reserva", min_value=0)
dias_lista_espera = st.number_input("Días en lista de espera", min_value=0)
adr = st.number_input("Tarifa diaria promedio (ADR)", min_value=0.0)
parqueo = st.number_input("Plazas de parqueo solicitadas", min_value=0)
solicitudes = st.number_input("Solicitudes especiales", min_value=0)

# Asignación de datos al DataFrame
if hotel == "Resort Hotel":
    input_data["hotel_Resort Hotel"] = 1
if f"comida_{comida}" in input_data.columns:
    input_data[f"comida_{comida}"] = 1
if f"pais_{pais}" in input_data.columns:
    input_data[f"pais_{pais}"] = 1
if f"segmento_mercado_{segmento}" in input_data.columns:
    input_data[f"segmento_mercado_{segmento}"] = 1
if f"canal_distribucion_{canal}" in input_data.columns:
    input_data[f"canal_distribucion_{canal}"] = 1
if f"tipo_habitacion_reservada_{habitacion_reservada}" in input_data.columns:
    input_data[f"tipo_habitacion_reservada_{habitacion_reservada}"] = 1
if f"tipo_habitacion_asignada_{habitacion_asignada}" in input_data.columns:
    input_data[f"tipo_habitacion_asignada_{habitacion_asignada}"] = 1
if f"tipo_deposito_{deposito}" in input_data.columns:
    input_data[f"tipo_deposito_{deposito}"] = 1
if f"tipo_cliente_{tipo_cliente}" in input_data.columns:
    input_data[f"tipo_cliente_{tipo_cliente}"] = 1

input_data["mes_fecha_llegada"] = mes
input_data["ano_fecha_llegada"] = ano
input_data["numero_semana_fecha_llegada"] = semana
input_data["dia_del_mes_fecha_llegada"] = dia
input_data["tiempo_espera"] = tiempo_espera
input_data["estancias_noches_semana"] = noches_semana
input_data["estancias_noches_fin_semana"] = noches_finde
input_data["adultos"] = adultos
input_data["ninos"] = ninos
input_data["bebes"] = bebes
input_data["es_huesped_repetido"] = 1 if repetido == "Sí" else 0
input_data["cancelaciones_previas"] = cancelaciones_previas
input_data["reservas_anteriores_no_canceladas"] = reservas_previas_ok
input_data["cambios_reserva"] = cambios
input_data["dias_en_lista_espera"] = dias_lista_espera
input_data["tarifa_diaria_promedio"] = adr
input_data["plazas_aparcamiento_solicitadas"] = parqueo
input_data["total_solicitudes_especiales"] = solicitudes

# Predicción
if st.button("Predecir Cancelación"):
    pred = model.predict(input_data)[0]
    st.subheader("Resultado:")
    if pred == 1:
        st.error("⚠️ La reserva probablemente será cancelada.")
    else:
        st.success("✅ La reserva probablemente NO será cancelada.")

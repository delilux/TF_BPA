import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Título de la app
st.set_page_config(page_title="Predicción de Cancelaciones de Reserva", layout="wide")
st.title("🔍 Predicción de Cancelación de Reservas de Hotel")

# Cargar el modelo y las columnas
@st.cache_resource
def load_model():
    model = joblib.load("modelo_xgboost_hiperparametrizado.pkl")
    columnas_modelo = joblib.load("columnas_modelo.pkl")
    return model, columnas_modelo

model, columnas_modelo = load_model()

# Campos del formulario que el usuario puede modificar
with st.form("formulario_reserva"):
    st.subheader("📝 Completa los datos relevantes de la reserva")
    
    hotel = st.selectbox("Tipo de hotel", ["Resort Hotel", "City Hotel"])
    comida = st.selectbox("Tipo de comida", ["FB", "HB", "SC", "Undefined"])
    pais = st.selectbox("País de origen", ["PER", "BRA", "ESP", "USA", "FRA"])  # puedes ampliar esta lista
    segmento = st.selectbox("Segmento de mercado", ["Online TA", "Offline TA/TO", "Groups", "Direct"])
    canal = st.selectbox("Canal de distribución", ["TA/TO", "Direct", "GDS", "Undefined"])
    habitacion_reservada = st.selectbox("Tipo de habitación reservada", ["D", "E", "F", "G"])
    habitacion_asignada = st.selectbox("Tipo de habitación asignada", ["D", "E", "F", "G"])
    tipo_deposito = st.selectbox("Tipo de depósito", ["Non Refund", "Refundable"])
    tipo_cliente = st.selectbox("Tipo de cliente", ["Transient", "Transient-Party", "Group"])

    mes = st.slider("Mes de llegada", 1, 12, 7)
    ano = st.slider("Año de llegada", 2018, 2025, 2025)
    semana = st.slider("Número de semana", 1, 52, 30)
    dia = st.slider("Día del mes", 1, 31, 15)

    adultos = st.number_input("Adultos", min_value=1, value=2)
    ninos = st.number_input("Niños", min_value=0, value=0)
    bebes = st.number_input("Bebés", min_value=0, value=0)

    tarifa = st.number_input("Tarifa diaria promedio (ADR)", min_value=0.0, value=100.0)
    estancias_semana = st.slider("Estancias entre semana", 0, 10, 3)
    estancias_fin_semana = st.slider("Estancias fin de semana", 0, 5, 2)

    tiempo_espera = st.number_input("Tiempo de espera (días)", min_value=0, value=10)
    solicitudes = st.slider("Total solicitudes especiales", 0, 5, 1)
    cambios = st.slider("Cambios de reserva", 0, 5, 0)
    cancelaciones_previas = st.slider("Cancelaciones previas", 0, 50, 0)
    reservas_previas = st.slider("Reservas previas no canceladas", 0, 5, 0)
    parqueo = st.slider("N° de plazas de aparcamiento solicitadas", 0, 3, 1)
    lista_espera = st.slider("Días en lista de espera", 0, 10, 0)

    es_repetido = st.selectbox("¿Es huésped repetido?", ["No", "Sí"])

    submit = st.form_submit_button("Predecir cancelación")

# Procesar datos y predecir
if submit:
    # Crear DataFrame base con ceros
    input_data = pd.DataFrame(np.zeros((1, len(columnas_modelo))), columns=columnas_modelo)

    # Setear columnas categóricas
    input_data.loc[0, f'hotel_{hotel}'] = 1
    input_data.loc[0, f'comida_{comida}'] = 1
    input_data.loc[0, f'pais_{pais}'] = 1
    input_data.loc[0, f'segmento_mercado_{segmento}'] = 1
    input_data.loc[0, f'canal_distribucion_{canal}'] = 1
    input_data.loc[0, f'tipo_habitacion_reservada_{habitacion_reservada}'] = 1
    input_data.loc[0, f'tipo_habitacion_asignada_{habitacion_asignada}'] = 1
    input_data.loc[0, f'tipo_deposito_{tipo_deposito}'] = 1
    input_data.loc[0, f'tipo_cliente_{tipo_cliente}'] = 1

    # Setear columnas numéricas
    input_data.loc[0, "mes_fecha_llegada"] = mes
    input_data.loc[0, "ano_fecha_llegada"] = ano
    input_data.loc[0, "numero_semana_fecha_llegada"] = semana
    input_data.loc[0, "dia_del_mes_fecha_llegada"] = dia
    input_data.loc[0, "adultos"] = adultos
    input_data.loc[0, "ninos"] = ninos
    input_data.loc[0, "bebes"] = bebes
    input_data.loc[0, "tarifa_diaria_promedio"] = tarifa
    input_data.loc[0, "estancias_noches_semana"] = estancias_semana
    input_data.loc[0, "estancias_noches_fin_semana"] = estancias_fin_semana
    input_data.loc[0, "tiempo_espera"] = tiempo_espera
    input_data.loc[0, "total_solicitudes_especiales"] = solicitudes
    input_data.loc[0, "cambios_reserva"] = cambios
    input_data.loc[0, "cancelaciones_previas"] = cancelaciones_previas
    input_data.loc[0, "reservas_anteriores_no_canceladas"] = reservas_previas
    input_data.loc[0, "plazas_aparcamiento_solicitadas"] = parqueo
    input_data.loc[0, "dias_en_lista_espera"] = lista_espera
    input_data.loc[0, "es_huesped_repetido"] = 1 if es_repetido == "Sí" else 0

    # Realizar la predicción
    try:
        resultado = model.predict(input_data)[0]
        probabilidad = model.predict_proba(input_data)[0][1]
        if resultado == 1:
            st.error(f"🚫 La reserva probablemente **será cancelada** (probabilidad: {probabilidad:.2%})")
        else:
            st.success(f"✅ La reserva probablemente **NO será cancelada** (probabilidad: {probabilidad:.2%})")
    except Exception as e:
        st.exception(f"❌ Error al predecir: {str(e)}")

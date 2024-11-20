import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cargar el modelo, el codificador One-Hot y el escalador
model = joblib.load('Notebook/gradient_boosting_model.pkl')
one_hot_encoder = joblib.load('Notebook/one_hot_encoder.pkl')
scaler = joblib.load('Notebook/scaler.pkl')

# Crear la aplicación en Streamlit
st.title('Predicción de Transporte de Pasajeros')
st.write('Introduce los datos del pasajero para predecir si será transportado o no.')

# Crear formularios de entrada
home_planet = st.selectbox('Home Planet', ['Europa', 'Earth', 'Mars'])
cryo_sleep = st.selectbox('Cryo Sleep', ['True', 'False'])
destination = st.selectbox('Destination', ['TRAPPIST-1e', '55 Cancri e', 'PSO J318.5-22'])
age = st.slider('Age', 0, 100, 25)
vip = st.selectbox('VIP', ['True', 'False'])
room_service = st.number_input('Room Service', 0.0, 10000.0, 0.0)
food_court = st.number_input('Food Court', 0.0, 10000.0, 0.0)
shopping_mall = st.number_input('Shopping Mall', 0.0, 10000.0, 0.0)
spa = st.number_input('Spa', 0.0, 10000.0, 0.0)
vr_deck = st.number_input('VR Deck', 0.0, 10000.0, 0.0)

# Crear el dataframe de entrada
input_data = pd.DataFrame({
    'Age': [age],
    'RoomService': [room_service],
    'FoodCourt': [food_court],
    'ShoppingMall': [shopping_mall],
    'Spa': [spa],
    'VRDeck': [vr_deck],
    'TotalSpending': [room_service + food_court + shopping_mall + spa + vr_deck],
    'SpendingPerAge': [(room_service + food_court + shopping_mall + spa + vr_deck) / (age + 1)]
})

# Convertir las entradas categóricas a los mismos tipos que en el entrenamiento
input_data_categorical = pd.DataFrame({
    'HomePlanet': [home_planet],
    'CryoSleep': [cryo_sleep == 'True'],
    'Destination': [destination],
    'VIP': [vip == 'True']
})

# Añadir las variables categóricas codificadas
input_data = pd.concat([input_data, pd.DataFrame(one_hot_encoder.transform(input_data_categorical), columns=one_hot_encoder.get_feature_names_out(['HomePlanet', 'CryoSleep', 'Destination', 'VIP']))], axis=1)

# Escalar los datos de entrada
input_data_scaled = scaler.transform(input_data)

# Hacer la predicción
prediction = model.predict(input_data_scaled)

# Mostrar el resultado
st.write('Predicción:', 'Transportado' if prediction[0] else 'No Transportado')
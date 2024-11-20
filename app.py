import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Cargar el modelo
model = joblib.load('Notebook/gradient_boosting_model.pkl')

# Cargar el codificador One-Hot y el escalador
one_hot_encoder = joblib.load('Notebook/one_hot_encoder.pkl')
scaler = joblib.load('Notebook/scaler.pkl')

# Título de la aplicación
st.title('Spaceship Titanic Rescue Mission Prediction')

# Descripción de la aplicación
st.write("""
Esta aplicación predice si un pasajero fue transportado a una dimensión alternativa en la misión de rescate del Spaceship Titanic.
""")

# Función para hacer predicciones
def predict_transportation(data):
    prediction = model.predict(data)
    return prediction

# Crear un formulario para la entrada de datos del usuario
with st.form(key='prediction_form'):
    st.header('Ingrese los datos del pasajero:')
    
    HomePlanet = st.selectbox('HomePlanet', ['Earth', 'Europa', 'Mars'])
    CryoSleep = st.selectbox('CryoSleep', ['True', 'False'])
    Destination = st.selectbox('Destination', ['TRAPPIST-1e', '55 Cancri e', 'PSO J318.5-22'])
    VIP = st.selectbox('VIP', ['True', 'False'])
    RoomService = st.number_input('RoomService', min_value=0, value=0)
    FoodCourt = st.number_input('FoodCourt', min_value=0, value=0)
    ShoppingMall = st.number_input('ShoppingMall', min_value=0, value=0)
    Spa = st.number_input('Spa', min_value=0, value=0)
    VRDeck = st.number_input('VRDeck', min_value=0, value=0)
    Age = st.number_input('Age', min_value=0, value=0)
    
    submit_button = st.form_submit_button(label='Predecir')

# Procesar la entrada del usuario y hacer predicciones
if submit_button:
    # Crear un DataFrame con los datos del usuario
    user_data = pd.DataFrame({
        'HomePlanet': [HomePlanet],
        'CryoSleep': [CryoSleep],
        'Destination': [Destination],
        'VIP': [VIP],
        'RoomService': [RoomService],
        'FoodCourt': [FoodCourt],
        'ShoppingMall': [ShoppingMall],
        'Spa': [Spa],
        'VRDeck': [VRDeck],
        'Age': [Age]
    })
    
    # Realizar la codificación One-Hot para las variables categóricas
    user_data_encoded = one_hot_encoder.transform(user_data[['HomePlanet', 'CryoSleep', 'Destination', 'VIP']])
    
    # Crear nuevas características
    user_data['TotalSpending'] = user_data['RoomService'] + user_data['FoodCourt'] + user_data['ShoppingMall'] + user_data['Spa'] + user_data['VRDeck']
    user_data['SpendingPerAge'] = user_data['TotalSpending'] / (user_data['Age'] + 1)  # +1 para evitar división por cero
    
    # Escalar los datos
    user_data_scaled = scaler.transform(user_data.drop(['HomePlanet', 'CryoSleep', 'Destination', 'VIP'], axis=1))
    
    # Concatenar las características codificadas y escaladas
    user_data_final = np.hstack((user_data_encoded, user_data_scaled))
    
    # Hacer la predicción
    prediction = predict_transportation(user_data_final)
    
    # Mostrar el resultado
    if prediction[0] == 1:
        st.success('El pasajero fue transportado a una dimensión alternativa.')
    else:
        st.success('El pasajero no fue transportado a una dimensión alternativa.')
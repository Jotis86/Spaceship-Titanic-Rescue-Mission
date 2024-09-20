import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier

# Cargar los datos
TRAIN_CSV = "C:/Users/juane/OneDrive/Escritorio/Datos/Kaggle_Titanic/train.csv"
TEST_CSV = "C:/Users/juane/OneDrive/Escritorio/Datos/Kaggle_Titanic/test.csv"
train_df = pd.read_csv(TRAIN_CSV, usecols=['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported'])
test_df = pd.read_csv(TEST_CSV, usecols=['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'])
all_data_df = pd.concat([train_df, test_df], keys=['train', 'test'])

# Manejar valores faltantes
all_data_df['Age'].fillna(all_data_df['Age'].mean(), inplace=True)
all_data_df['RoomService'].fillna(all_data_df['RoomService'].mean(), inplace=True)
all_data_df['FoodCourt'].fillna(all_data_df['FoodCourt'].mean(), inplace=True)
all_data_df['ShoppingMall'].fillna(all_data_df['ShoppingMall'].mean(), inplace=True)
all_data_df['Spa'].fillna(all_data_df['Spa'].mean(), inplace=True)
all_data_df['VRDeck'].fillna(all_data_df['VRDeck'].mean(), inplace=True)
all_data_df['HomePlanet'].fillna(all_data_df['HomePlanet'].mode()[0], inplace=True)
all_data_df['CryoSleep'].fillna(all_data_df['CryoSleep'].mode()[0], inplace=True)
all_data_df['Destination'].fillna(all_data_df['Destination'].mode()[0], inplace=True)
all_data_df['VIP'].fillna(all_data_df['VIP'].mode()[0], inplace=True)
all_data_df['Transported'].fillna(all_data_df['Transported'].mode()[0], inplace=True)

# Feature Engineering
all_data_df['TotalSpending'] = all_data_df['RoomService'] + all_data_df['FoodCourt'] + all_data_df['ShoppingMall'] + all_data_df['Spa'] + all_data_df['VRDeck']
all_data_df['SpendingPerAge'] = all_data_df['TotalSpending'] / (all_data_df['Age'] + 1)

# One-Hot Encoding
one_hot_encoder = OneHotEncoder(drop='first', sparse_output=False)
categorical_columns = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
encoded_features = one_hot_encoder.fit_transform(all_data_df[categorical_columns])
encoded_df = pd.DataFrame(encoded_features, columns=one_hot_encoder.get_feature_names_out(categorical_columns))
all_data_df = all_data_df.reset_index(drop=True)
all_data_df = pd.concat([all_data_df, encoded_df], axis=1)

# Escalar los datos
scaler = MinMaxScaler()
features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpending', 'SpendingPerAge'] + list(encoded_df.columns)
X = all_data_df[features]
X_scaled = scaler.fit_transform(X)

# Dividir los datos
X_train = X_scaled[:len(train_df)]
y_train = train_df['Transported'].astype(int)
X_test = X_scaled[len(train_df):]

# Verificar el tamaño de los datos de entrenamiento
st.write(f"Tamaño de X_train: {X_train.shape}")
st.write(f"Tamaño de y_train: {y_train.shape}")

# Entrenar el modelo
gb_clf = GradientBoostingClassifier(random_state=42)
gb_clf.fit(X_train, y_train)

# Verificar la precisión del modelo en los datos de entrenamiento
train_accuracy = gb_clf.score(X_train, y_train)
st.write(f"Precisión del modelo en los datos de entrenamiento: {train_accuracy}")

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
input_data = pd.concat([input_data, pd.DataFrame(one_hot_encoder.transform(input_data_categorical), columns=one_hot_encoder.get_feature_names_out(categorical_columns))], axis=1)

# Escalar los datos de entrada
input_data_scaled = scaler.transform(input_data)

# Hacer la predicción
prediction = gb_clf.predict(input_data_scaled)

# Mostrar el resultado
st.write('Predicción:', 'Transportado' if prediction[0] else 'No Transportado')



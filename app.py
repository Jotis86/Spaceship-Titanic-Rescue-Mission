import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn

# Verificar versiones de las bibliotecas
st.write(f"scikit-learn version: {sklearn.__version__}")
st.write(f"joblib version: {joblib.__version__}")

# Load the model, One-Hot encoder, and scaler
try:
    model = joblib.load('Notebook/gradient_boosting_model.pkl')
    one_hot_encoder = joblib.load('Notebook/one_hot_encoder.pkl')
    scaler = joblib.load('Notebook/scaler.pkl')
except Exception as e:
    st.error(f"Error loading model or encoders: {e}")

# Title and main image
st.title('🚀 Passenger Transport Prediction 🌌')
st.image('images/Cat_2.png')

# Sidebar with image and navigation buttons
st.sidebar.image('images/nave.png', use_column_width=True)
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['Project Objectives', 'Development Process', 'Tools Used', 'Visualizations', 'Predictions'])

if page == 'Project Objectives':
    st.header('🎯 Project Objectives')
    st.write("""
    Welcome to the Passenger Transport Prediction project for the Spaceship Titanic rescue mission. 
    The goal of this project is to predict whether a passenger was transported to an alternate dimension using 
    machine learning techniques. This project is part of a Kaggle challenge to help save the passengers and change history.
    
    In the year 2912, the Spaceship Titanic, an interstellar passenger liner, was launched with nearly 13,000 passengers. 
    It embarked on its maiden voyage, transporting emigrants from our solar system to three newly habitable exoplanets orbiting nearby stars.
    
    Unfortunately, while orbiting Alpha Centauri en route to its first destination, the Spaceship Titanic collided with a space-time anomaly hidden within a dust cloud. 
    As a result, nearly half of the passengers were transported to an alternate dimension. The mission now is to predict which passengers were transported using the records recovered from the ship's damaged computer system.
    """)

elif page == 'Development Process':
    st.header('🛤️ Development Process')
    st.write("""
    The development process for this project involved several key steps:
    
    1. **Import Libraries** 📚: Import the necessary libraries for data analysis and visualization.
    2. **Load Dataset** 📥: Load and combine the datasets.
    3. **Initial Exploration** 🔍: Display the first few rows and all columns of the combined dataset.
    4. **Data Cleaning** 🧹: Handle missing values and perform One-Hot encoding for categorical variables.
    5. **Feature Engineering** 🛠️: Create new features and scale the data.
    6. **Model Training** 🤖: Train the Gradient Boosting model and evaluate its performance.
    
    Each of these steps was crucial in building a robust model capable of accurately predicting whether a passenger was transported.
    """)

elif page == 'Tools Used':
    st.header('🛠️ Tools Used')
    st.write("""
    The following tools and libraries were used in this project:
    
    - **Python** 🐍: Programming language
    - **Pandas** 🐼: Data manipulation
    - **NumPy** 🔢: Numerical computations
    - **Matplotlib & Seaborn** 📊: Data visualization
    - **Scikit-learn** 🤖: Machine learning
    - **Streamlit** 🌐: Web application framework
    
    These tools provided the necessary functionality to preprocess the data, train the model, and build the web application.
    """)

elif page == 'Visualizations':
    st.header('📊 Visualizations')
    
    # Display images stored in the images folder
    st.subheader('Age Distribution 🎂')
    st.image('images/Age.png')
    st.write("""
    This visualization shows the distribution of ages among the passengers. 
    It helps to understand the age demographics of the passengers on the Spaceship Titanic.
    """)
    
    st.subheader('Age Distribution by Transported Status 🚀')
    st.image('images/Age_d.png')
    st.write("""
    This visualization shows the age distribution of passengers based on their transported status. 
    It helps to identify any patterns or trends in the age groups that were more likely to be transported.
    """)
    
    st.subheader('Service Spending Distribution 💸')
    st.image('images/Services.png')
    st.write("""
    This visualization shows the distribution of spending on various services such as RoomService, FoodCourt, ShoppingMall, Spa, and VRDeck. 
    It helps to understand the spending behavior of the passengers.
    """)
    
    st.subheader('Age Distribution by Home Planet 🌍')
    st.image('images/Age by planet.png')
    st.write("""
    This visualization shows the age distribution of passengers based on their home planet. 
    It helps to identify any differences in age demographics among passengers from different planets.
    """)
    
    st.subheader('Feature Importance in the Model 🔍')
    st.image('images/features.png')
    st.write("""
    This visualization shows the importance of various features in the Gradient Boosting model. 
    It helps to understand which features have the most significant impact on the prediction of whether a passenger was transported.
    """)

elif page == 'Predictions':
    st.header('🔮 Predictions')
    
    # Create input forms
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

    # Create the input dataframe
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

    # Convert categorical inputs to the same types as in training
    input_data_categorical = pd.DataFrame({
        'HomePlanet': [home_planet],
        'CryoSleep': [cryo_sleep == 'True'],
        'Destination': [destination],
        'VIP': [vip == 'True']
    })

    # Add encoded categorical variables
    input_data = pd.concat([input_data, pd.DataFrame(one_hot_encoder.transform(input_data_categorical), columns=one_hot_encoder.get_feature_names_out(['HomePlanet', 'CryoSleep', 'Destination', 'VIP']))], axis=1)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Make the prediction
    prediction = model.predict(input_data_scaled)

    # Display the result
    st.write('Prediction:', '🚀 Transported to another dimension! 🌌' if prediction[0] else '😔 Not transported, still here.')

# Custom GitHub button
st.sidebar.markdown("""
    <a href="https://github.com/Jotis86/Spaceship-Titanic-Rescue-Mission" target="_blank">
        <button style="background-color:blue;color:white;padding:10px;border-radius:5px;border:none;cursor:pointer;">
            GitHub Repo
        </button>
    </a>
""", unsafe_allow_html=True)
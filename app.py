#building the cloud app 
# transforming input
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


#streanlit page configurations
st.set_page_config(
    page_title="Data-Driven California Housing Predictor"
)
#1. Load the full end-to-end pipeline model
def load_california_model():
    model_path= r"C:\Users\Nnamdi Kelvin\Desktop\ml\linear_reg\Best_model(Gridsearch).pkl"
    try:
        if os.path.exists(model_path):
            #preprocessor = joblib.load(preprocessor_path)
            model = joblib.load(model_path)
            st.success("Model loaded successfully.")
            return model
        #else:
            
            #return None, None
    except Exception as e:
        st.error(f"Model file not found. Please ensure the model file is in same directory ...")
        st.write(type(e).__name__)
        return None, None

best_model = load_california_model()

#custom Transformer for the learning algorithms
# Adjust these indices to match your specific dataset column positions
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    # Fix: Ensure exactly TWO underscores on both sides of init
    def __init__(self, add_bedrooms_per_room=True): 
        self.add_bedrooms_per_room = add_bedrooms_per_room
        
    def fit(self, X, y=None):
        return self  # Nothing else to do here
        
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)


def load_preprocessor_transformer():
    preprocessor_path= r"C:\Users\Nnamdi Kelvin\Desktop\ml\linear_reg\preprocessor.pkl"

    try:
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
            st.success("Transformer Loaded successfully..")
        return preprocessor
    except Exception as e:
        st.error(f"Transformers  file not found. Please ensure the file is in same directory ...")
        st.write(type(e).__name__)
        return None, None

best_preprocessor = load_preprocessor_transformer()

#location mapping
California_profile = {
    '<1H OCEAN': {
        'longitude': -118.85816820118181, 
        'latitude': 34.5697911227154, 
        'population': 1517.605469286794, 
        'households': 517.3799642709907
        }, 
     'INLAND': {
         'longitude': -119.73583618396047, 
         'latitude': 36.73573166096541, 
         'population': 1383.1854808057772, 
         'households': 473.7793614595211
         }, 
     'ISLAND': {
         'longitude': -118.405, 
         'latitude': 33.385000000000005, 
         'population': 761.0, 
         'households': 302.0
         }, 
     'NEAR BAY': {
         'longitude': -122.25996210070385, 
         'latitude': 37.80126150514347, 
         'population': 1212.5381700054143, 
         'households': 480.0357336220899
         }, 
     'NEAR OCEAN': {
         'longitude': -119.30370527306968, 
         'latitude': 34.70944915254237, 
         'population': 1355.3959510357815, 
         'households': 499.7306967984934
         }
}
st.set_page_config(page_title="California Housing Price Prediction",layout="centered")
st.title("Data-Driven California Home Value Estimator")
st.write("This application scales individual user inputs up using true baseline metrics extracted from the census dataset.")


st.markdown("---")

#3. Simple User Inputs


def get_user_input():
    """ Takes individual user preference and maps them to census district-level
    statistics using true dataFrame baseline averages.
    returns a predictions"""

    ocean_proximity = st.sidebar.selectbox(
    "What Type of  environment are you looking for?",
    list(California_profile.keys())
)
    user_rooms =  st.sidebar.number_input("Total Rooms desired in a house", min_value=1, max_value=15, value=5)
    user_bedrooms =  st.sidebar.number_input("Total BedRooms desired", min_value=1, max_value=8, value=2)
    user_age = st.sidebar.slider("Property's Age(Years)", min_value=1, max_value=52, value=25)

    user_income =st.sidebar.number_input("Your Annual Household Income(in USD)", min_value=10000, max_value=250000, value=75000, step=5000)
    st.caption("Example: $75,000 maps to data income of 7.5")

    housing_metrics = California_profile[ocean_proximity]

    true_avg_households = housing_metrics["households"]
    true_avg_population = housing_metrics["population"]

    #MATHS Transformers: scaling individual values to match census blocks
    calculated_total_rooms = user_rooms * true_avg_households
    calculated_total_bedroom = user_bedrooms * true_avg_households
    scaled_income = user_income / 10000.0

    #Pack into exact structural layout for model input
    User_input = {
        "longitude": [housing_metrics["longitude"]],
        "latitude": [housing_metrics["latitude"]],
        "housing_median_age": [user_age],
        "total_rooms": [calculated_total_rooms],
        "total_bedrooms": [calculated_total_bedroom],
        "population":[true_avg_population],
        "households": [true_avg_households],
        "median_income": [scaled_income],
        "ocean_proximity" : [ocean_proximity]

    }
    return pd.DataFrame(User_input), ocean_proximity


#streamlit UI Layout Setup
st.set_page_config(page_title="Data-Driven California Housing Predictor", layout="centered")

st.title("Data-Driven California Home Value Estimator.")
st.write("This application scales users input baseline metrics extracted from the census datasets.")

#centre title
st.markdown("<h1 stlye='text-align: center;'>California Home Value Estimator App</h1>", unsafe_allow_html=True)

#Call the user input fuction to display fields and get the input DataFrame
left_column, right_column = st.columns(2)

with right_column:
    st.header("Predict home Value")
    #users input from sidebar

    user_data, choosen_ocean_type = get_user_input()


# Prediction Execution

if st.button("Estimate Property value", type="primary",  use_container_width=True):
    with st.spinner("Processing Through  your model pipeline..."):
        try:
            # transformed_features handles All scaling, encoding, and features engineering perfectly
            transformed_features = best_preprocessor.transform(user_data)
            predictions = best_model.predict(transformed_features)
            predicted_value = float(predictions[0])

            st.success(f" Date-Backed Estimated Value: ${predicted_value:,.2f}")
            st.caption(f"Calculated  utilising actual dataset baseline averages for a California {choosen_ocean_type} neighbour's block")

        except Exception as e:
            st.error(f"Prediction Error: {e}")
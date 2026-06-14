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
    page_title="California Housing Price Prediction",
    layout="wide",
)


#st.markdown("---")
#custom style for the app

st.markdown("""
    <style>
    .prediction-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 12px;
        border-left: 6px solid #1e3c72;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        margin-top: 15px;
        margin-bottom: 15px;
    }
    </style>""", unsafe_allow_html=True)
#-----------------------------------------------------------------------------------------------------
# Step 1 load custom Transformer and the model pipeline(Load the full end-to-end pipeline model)
#----------------------------------------------------------------------------------------------------

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


#load model from base dir
def load_california_model():
    base_dir = os.path.dirname(__file__)
    model_path= os.path.join(base_dir, "Models", "Best_model(Gridsearch).pkl")
    try:
        if os.path.exists(model_path):
            #preprocessor = joblib.load(preprocessor_path)
            model = joblib.load(model_path)
            st.sidebar.success("Model loaded successfully.")
            return model
        #else:
            
            #return None, None
    except Exception as e:
        st.error(f"Model file not found. Please ensure the model file is in same directory ...")
        st.write(type(e).__name__)
        return None, None



#load the preprocessor

def load_preprocessor_transformer():
    base_dir = os.path.dirname(__file__)
    preprocessor_path= os.path.join(base_dir, "Models", "preprocessor.pkl")

    try:
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
            st.sidebar.success("Transformer Loaded successfully..")
        return preprocessor
    except Exception as e:
        st.error(f"Transformers  file not found. Please ensure the file is in same directory ...")
        st.write(type(e).__name__)
        return None, None

#load my assets  for predictions
best_model = load_california_model()
best_preprocessor = load_preprocessor_transformer()


#-----------------------------------------------------------------------------------------
# Step 2: location maping and dictionary
#-----------------------------------------------------------------------------------------


#location mapping
California_profile = {
    '<1H OCEAN'     : { 'longitude': -118.8581, 'latitude': 34.5697, 'population': 1517.6054, 'households': 517.3799}, 
     'INLAND'       : {'longitude': -119.7358, 'latitude': 36.7357, 'population': 1383.1854, 'households': 473.7793}, 
     'ISLAND'       : {'longitude': -118.405, 'latitude': 33.3850, 'population': 761.0, 'households': 302.0}, 
     'NEAR BAY'     : {'longitude': -122.2599, 'latitude': 37.8012, 'population': 1212.5381, 'households': 480.0357}, 
     'NEAR OCEAN'   : {'longitude': -119.3037, 'latitude': 34.7094, 'population': 1355.3959, 'households': 499.7306}
}

#-------------------------------------------------------------------------------------------
#3. Simple User Inputs
#------------------------------------------------------------------------------------


def get_user_input():
    """ Takes individual user preference and maps them to census district-level
    statistics using true dataFrame baseline averages.
    returns a predictions"""
    st.sidebar.image(
        "https://images.unsplash.com/photo-1560518883-ce09059eeffa?w=500&auto=format&fit=crop&q=60", 
        width="stretch"
    )
    st.sidebar.markdown("Evaluation Configuration")
    

    ocean_proximity = st.sidebar.selectbox(
    "What Type of  environment are you looking for?",
    list(California_profile.keys())
)
    user_rooms =  st.sidebar.number_input("Total Rooms desired in a House", min_value=1, max_value=15, value=5)
    user_bedrooms =  st.sidebar.number_input("Bed Rooms desired in a House", min_value=1, max_value=8, value=2)
    user_age = st.sidebar.slider("Property's Operstional Age(Years)", min_value=1, max_value=52, value=25)

    user_income =st.sidebar.number_input("Your Annual Household Income (in USD)", min_value=10000, max_value=250000, value=75000, step=5000)
    #st.caption("Example: $75,000 maps to data income of 7.5")

    # Pull census district group baseline averages based on region category

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

#---------------------------------------------------------------------
#streamlit UI Layout Setup (# STEP 4: MAIN DASHBOARD PRESENTER LAYOUT)
#---------------------------------------------------------------------
st.image(
        "headerpicture.jpg",
        width="stretch"
    )

st.markdown("<h1 style='text-align: center; color: #FDB515; margin-bottom: 5px;'> 🏡California Home Value Estimator App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #003262; margin-bottom: 25px;'>This application scales individual user inputs up using true baseline metrics extracted from the census dataset.</p>", unsafe_allow_html=True)

#Call the user input fuction to display fields and get the input DataFrame
left_column, right_column = st.columns(2)


with left_column:
    """I will put an interactive chart here"""

with right_column:
    st.header("Predict home Value")
    #users input from sidebar
    st.subheader(" Target Assessment Pipeline")
    # Call input pipeline to instantiate core input form fields on sidebar space
    st.info("Adjust parameter configuration dimensions inside the left sidebar navigation panel to re-evaluate structural predictions live.")
    user_data, choosen_ocean_type = get_user_input()

st.markdown("---")

#---------------------------------------------------------------------------------------------------------------
#Step 5: predictions and interactive maps
# Prediction Execution
#----------------------------------------------------------------------------------------------------------------

if st.button("Estimate Property value", type="primary", width="stretch"):
    with st.spinner("Processing Through  your model pipeline..."):
        try:
            if best_preprocessor is None or best_model is None:
                st.error("Assets Error: Pipeline files are currently unlinked, corrupt, or missing from local machine arrays.")
            else:
                 # transformed_features handles All scaling, encoding, and features engineering perfectly
                transformed_features = best_preprocessor.transform(user_data)
                predictions = best_model.predict(transformed_features)
                predicted_value = float(predictions[0])
                
                # 2. Open a dynamic side-by-side grid panel to host the pricing results vs. map layout
                result_display, map_display = st.columns([1, 1], gap="large")
                with result_display:
                    st.markdown( f"""
                        <div class="prediction-card">
                            <p style="margin:0; font-size:12px; text-transform:uppercase; color:#64748b; font-weight:600;">System Execution State</p>
                            <h2 style="margin:5px 0 0 0; color:#1e3c72; font-size: 28px;"> Estimated Value: ${predicted_value:,.2f}</h2>
                            <p style="margin:10px 0 0 0; font-size:13px; color:#475569; line-height:1.5;">
                                Calculated utilising baseline spatial records for a California <strong>{choosen_ocean_type}</strong> neighborhood environment block.
                            </p>
                        </div>
                    """, unsafe_allow_html =True
                    )
                    st.success("Consistency constraints check verified successfully.")

                with map_display:
                    st.markdown("Census Target Spatial Coordinates Location")
                    
                    # 3. Package dynamic location coordinates array for st.map extraction layers
                    coordinates_df = pd.DataFrame({
                        'latitude': [float(user_data['latitude'][0])],
                        'longitude': [float(user_data['longitude'][0])]
                    })
                    
                    # Output responsive, interactive web map
                    st.map(coordinates_df, zoom=6, width="stretch")


            #st.success(f" Date-Backed Estimated Value: ${predicted_value:,.2f}")
            #st.caption(f"Calculated  utilising actual dataset baseline averages for a California {choosen_ocean_type} neighbour's block")

        except Exception as e:
            st.error(f"Prediction Error: {e}")
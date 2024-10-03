
import numpy as np
import pickle
import streamlit as st

# Load the saved model
loaded_model = pickle.load(open('C:/Users/rajkr/Downloads/solar_power_model (1).sav', 'rb'))

# Function for prediction
def solarpowergeneration_prediction(input_data):
    # Convert the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Make the prediction using the loaded model
    prediction = loaded_model.predict(input_data_reshaped)

    # Return the prediction result
    return prediction[0]

def main():
    # Title of the app
    st.title('Solar Power Generation Prediction Web App')

    # Getting input data from the user
    distance_to_solar_noon = st.text_input('Distance-to-Solar-noon Value (in radians)')
    temperature = st.text_input('Temperature (in degrees Celsius)')
    wind_direction = st.text_input('Wind Direction (in degrees)')
    wind_speed = st.text_input('Wind Speed (in meters per second)')
    sky_cover = st.text_input('Sky Cover')
    visibility = st.text_input('Visibility (in kilometers)')
    humidity = st.text_input('Humidity (in percentage)')
    average_wind_speed = st.text_input('Average Wind Speed (in meters per second)')
    average_pressure = st.text_input('Average Pressure (in mercury inches)')

    # Variable to store the prediction result
    result_generation = ''

    # Button for prediction
    if st.button('Solar Power Generation Test Result'):
        try:
            # Convert inputs to a list of floats
            input_data = [float(distance_to_solar_noon), float(temperature), float(wind_direction), 
                          float(wind_speed), float(sky_cover), float(visibility), float(humidity), 
                          float(average_wind_speed), float(average_pressure)]
            
            # Get prediction result
            result_generation = solarpowergeneration_prediction(input_data)
            st.success(f'Predicted Solar Power Generation: {result_generation}')
        except ValueError:
            st.error('Please enter valid numeric values for all inputs.')

# Run the main function
if __name__ == '__main__':
    main()
   
    
    

     







































































from numpy import byte, string_
import pandas as pd
from Tools import geocoder_here
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from PIL import Image
import streamlit as st
import datetime
import requests

st.write("""
# Fare Prediction in Newyork City
 Predict New York City fare using Random Forest Regressor and python!
""")
image = Image.open('ny.jpeg')
st.image(image, caption='By meshack kipsang',use_column_width=True)

#Get the data
df = pd.read_csv("newyorkfare.csv",nrows=400000)
st.subheader('Data Information:')
#Show the data as a table (you can also use st.write(df))
st.dataframe(df)
#Get statistics on the data
st.write(df.describe())
# Show the data as a chart.
chart = st.line_chart(df.astype(str))

#Split the data into independent 'X' and dependent 'Y' variables
X = df.iloc[:, 0:8].values
Y= df.iloc[:,-1].values
# Split the dataset into 80% Training set and 20% Testing set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Get the feature input from the user
map_df = pd.DataFrame({
    "lat": [40.71782, 40.73451],
    "lon": [-74.00547, -73.99853]
    })



st.sidebar.markdown("_Fill the information form below to get a New York City Taxi Fare Prediction_")
st.sidebar.markdown("### Datetime")

### Request Date of the course ###
date_col1, date_col2  = st.sidebar.beta_columns([1.75,1])
with date_col1:
    d = st.date_input("Date :", datetime.datetime.now())

### Request Time of the course ###
with date_col2:
    t = st.time_input('Time :', datetime.datetime.now())

### Format the datetime for our API ###
pickup_datetime = f"{d} {t} UTC"

### Display the Pickup Datetime ###
#st.sidebar.success(f'Pickup datetime : {pickup_datetime}')

### Request the Pickup location ###
st.sidebar.markdown("### Pickup Location")
pickup_adress = st.sidebar.text_input("Please enter the pickup address", "Central Park, NewYork")

### Getting Coordinates from Address locations ###
error1 = ""

try:
    pickup_coords = geocoder_here(pickup_adress)
except IndexError:
    error1 = "Pickup Address invalide, default coordinates : "
    pickup_coords = {
        "latitude": 40.78392,
        "longitude": -73.96584
    }


pickup_latitude = pickup_coords['latitude']
pickup_longitude = pickup_coords['longitude']
map_df.loc[0, "lat"] = float(pickup_latitude)
map_df.loc[0, "lon"] = float(pickup_longitude)

### Displaying the Pickup Coordinates ###
if error1 == "":
    st.sidebar.success(f'Lat: {pickup_latitude}, Lon : {pickup_longitude}')
else:
    st.sidebar.error(f'"{pickup_adress}" {error1} \n Lat : {pickup_latitude}, Lon : {pickup_longitude}')


### Request the Dropoff location ###
st.sidebar.markdown("### Dropoff Location")
dropoff_address = st.sidebar.text_input("Please enter the dropoff address", "JFK, NewYork")

### Getting Coordinates from Address locations ###
error2 = ""
try:
    dropoff_coords = geocoder_here(dropoff_address)
except IndexError:
    error2 = "Dropoff Address invalide, default coordinates : "
    dropoff_coords = {
        "latitude": 40.65467,
        "longitude": -73.78911
    }

dropoff_latitude = dropoff_coords['latitude']    
dropoff_longitude = dropoff_coords['longitude']

map_df.loc[1, "lat"] = float(dropoff_latitude)
map_df.loc[1, "lon"] = float(dropoff_longitude)

### Displaying the Pickup Coordinates ###
if error2 == "":
    st.sidebar.success(f'Lat : {dropoff_latitude}, Lon : {dropoff_longitude}')
else:
    st.sidebar.error(f'"{dropoff_address}" {error2} Lat: {dropoff_latitude}, Lon: {dropoff_longitude}')

### Request the Passenger Count ###
st.sidebar.markdown("### Passengers")
passenger_count = st.sidebar.slider('Please enter number of passengers', 1, 9, 1)

### Launch Fare Prediction ###
st.sidebar.markdown("### Prediction")
if st.sidebar.button('Get Fare Prediction'):

    params = {
        "key" : str(pickup_datetime),
        "pickup_datetime" : str(pickup_datetime),
        "pickup_longitude": float(pickup_longitude),
        "pickup_latitude": float(pickup_latitude),
        "dropoff_longitude" : float(dropoff_longitude),
        "dropoff_latitude": float(dropoff_latitude),
        "passenger_count": int(passenger_count)
    }   
    local_api_url = f"http://127.0.0.1:8000/predict_fare"
    cloud_url = "https://predict-api-vwdzl6iuoa-ew.a.run.app/predict_fare"
    response = requests.get(
    url=cloud_url, params=params
    ).json()

    st.info(f"Taxi Fare Predication from {pickup_adress} to {dropoff_address} : {round(response['prediction'], 2)}$ 🎉")

st.map(data=map_df, use_container_width=False)

RandomForestRegressor = RandomForestRegressor()
RandomForestRegressor.fit(X_train, Y_train)

#Show the models metrics
st.subheader('Model Test Root Mean Squared Error')
st.write( str(mean_squared_error(Y_test, RandomForestRegressor.predict(X_test) )))
prediction = RandomForestRegressor.predict(X_test)
st.subheader('Prediction: ')
st.write(prediction)

import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder,StandardScaler
import numpy as np

def get_clean_data():
    data=pd.read_csv("weather_classification_data.csv")
    label_encoder=LabelEncoder()
    data["Cloud Cover"]=label_encoder.fit_transform(data["Cloud Cover"])
    data["Season"]=label_encoder.fit_transform(data["Season"])
    data["Location"]=label_encoder.fit_transform(data["Location"])
    data["Weather Type"]=label_encoder.fit_transform(data["Weather Type"])
    return data

def add_sidebar():
    
    st.sidebar.header("Weather measurements")
    
    data=get_clean_data()

    slider_labels=[
        ("Temperature","Temperature"),("Humidity","Humidity"),
        ("Wind Speed","Wind Speed"),("Precipitation (%)","Precipitation (%)"),
        ("Cloud Cover","Cloud Cover"),("Atmospheric Pressure","Atmospheric Pressure"),
        ("UV Index","UV Index"),("Season","Season"),("Visibility (km)","Visibility (km)"),
        ("Location","Location")      
        
        ]
    
    input_dict={}
    
    for label,key in slider_labels:
        input_dict[key]=st.sidebar.slider(label,
                                          min_value=float(0),
                                          max_value=float(data[key].max()))
    return input_dict

def add_scaled_data(input_dict):
    data=get_clean_data()
    X=data.drop(["Weather Type"],axis=1)
    scaled_dict={}
    for key, value in input_dict.items():
        sc=StandardScaler()
        scaled_value=sc.fit_transform(X)
        scaled_dict[key]= scaled_value
    return scaled_dict


def add_predictions(input_data):
    model=pickle.load(open("model.pkl","rb"))
    scaler=pickle.load(open("scaler.pkl","rb"))
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    
    input_array_scaled = scaler.transform(input_array)
  
    if st.button("Predict"):
        prediction = model.predict(input_array_scaled)[0]
        
        st.write(f"Predicted class: {prediction}")
        
        




def main():
    st.set_page_config(page_title="Weather Classification ML App",layout="wide",
                       initial_sidebar_state="expanded")
    input_data=add_sidebar()
    
    with st.container():
        st.title("Weather classification app")
        st.write("This app is used for the prediction of the weather classification")
        
    add_predictions(input_data)
    
    
    
if __name__=="__main__":
    main()


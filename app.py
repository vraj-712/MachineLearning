import streamlit as st 
import pickle
import numpy as np 
import pandas as pd
fileName = "Model.sav"
loaded_model = pickle.load(open(fileName,"rb"))
def main():
    def loadDataset():
        heartData =pd.read_csv("heart_2020_cleaned.csv")
        return heartData 
    def userInterFace():
        
        option  = ["Yes","No"]
        option.sort()
        dict = {value:index for index, value in enumerate(option)}

        Race_value = heart["Race"].unique()
        Race_value.sort()
        Race_value_to_index = {value:index for index, value in enumerate(Race_value)}
        selected_Race_value = st.sidebar.selectbox("Race : ",Race_value,key=6)
        selected_Race_value_index = Race_value_to_index[selected_Race_value]
        
        Sex_value = heart["Sex"].unique()
        Sex_value.sort()
        Sex_value_to_index = {value:index for index, value in enumerate(Sex_value)}
        selected_Sex_value = st.sidebar.selectbox("Sex : ",Sex_value,key=5)
        selected_Sex_value_index = Sex_value_to_index[selected_Sex_value]
        
        age_value = st.sidebar.number_input("Age : ",0.0,150.0)
        
        bmi_value = st.sidebar.number_input("BMI Value")
        
        GenHealth_value = heart["GenHealth"].unique()
        GenHealth_value.sort()
        GenHealth_value_to_index = {value:index for index, value in enumerate(GenHealth_value)}
        selected_GenHealth_value = st.sidebar.selectbox("How can you define your general health?",GenHealth_value,key=9)
        selected_GenHealth_value_index = GenHealth_value_to_index[selected_GenHealth_value]

        physical_health = st.sidebar.number_input("For how many days during the past 30 days was your physical health not good?:",0.0,30.0)

        mental_health = st.sidebar.number_input("For how many days during the past 30 days was your mental health not good?:",0.0,30.0)

        selected_smoking_value = st.sidebar.selectbox("Do you smoke?",option,key=1)
        selected_smoking_value_index = dict[selected_smoking_value]
        
        selected_AlcoholDrinking_value = st.sidebar.selectbox("Do you consume alcohol?",option,key=2)
        selected_AlcoholDrinking_value_index = dict[selected_AlcoholDrinking_value]
        
        selected_Stroke_value = st.sidebar.selectbox("Did you have a stroke?",option,key=3)
        selected_Stroke_value_index = dict[selected_Stroke_value]
        
        selected_DiffWalking_value = st.sidebar.selectbox("Do you have serious difficulty walking or climbing stairs?",option,key=4)
        selected_DiffWalking_value_index = dict[selected_DiffWalking_value]
        
        Diabetic_value = heart["Diabetic"].unique()
        Diabetic_value.sort()
        Diabetic_value_to_index = {value:index for index, value in enumerate(Diabetic_value)}
        selected_Diabetic_value = st.sidebar.selectbox("Have you ever had diabetes?",Diabetic_value,key=7)
        selected_Diabetic_value_index = Diabetic_value_to_index[selected_Diabetic_value]
        
        selected_PhysicalActivity_value = st.sidebar.selectbox("Have you played any sports (running, biking, etc.) in the past month?(Regular)",option,key=8)
        selected_PhysicalActivity_value_index = dict[selected_PhysicalActivity_value]
        
        
        selected_Asthma_value = st.sidebar.selectbox("Do you have asthma?",option,key=10)
        selected_Asthma_value_index = dict[selected_Asthma_value]

        selected_KidneyDisease_value = st.sidebar.selectbox("Do you have kidney disease?",option,key=11)
        selected_KidneyDisease_value_index = dict[selected_KidneyDisease_value]
        
        selected_SkinCancer_value = st.sidebar.selectbox("Do you have skin cancer?",option,key=12)
        selected_SkinCancer_value_index = dict[selected_SkinCancer_value]
        
        sleep_value = st.sidebar.number_input("Average Sleep Time(In Last 30 Days) :",0.0,24.0,8.0)
        features  =pd.DataFrame( {"BMI":[bmi_value],
                      "Smoking":[selected_smoking_value_index],
                      "AlcoholDrinking":[selected_AlcoholDrinking_value_index],
                      "Stroke":[selected_Stroke_value_index],
                      "PhysicalHealth":[physical_health],
                      "MentalHealth":[mental_health],
                      "DiffWalking":[selected_DiffWalking_value_index],
                      "Sex":[selected_Sex_value_index],
                      "AgeCategory":[age_value],
                      "Race":[selected_Race_value_index],
                      "Diabetic":[selected_Diabetic_value_index],
                      "PhysicalActivity":[selected_PhysicalActivity_value_index],
                      "GenHealth":[selected_GenHealth_value_index],
                      "SleepTime":[sleep_value],
                      "Asthma":[selected_Asthma_value_index],
                      "KidneyDisease":[selected_KidneyDisease_value_index],
                      "SkinCancer":[selected_SkinCancer_value_index]})
        return features
        
        

    st.set_page_config(layout="wide")
    col1,col2 = st.columns([1,5])
    with col1:
        st.image("logo.png",width=150)
    with col2:
        st.title("Heart Disease Prediction Web App")
        st.subheader("""Assess Your Heart Health: Use this app to predict your risk of heart disease based on personal and medical factors.""")
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("OIP.png",
                 caption="Hi !! I am Dr. RFC.I'll help you diagnose your heart health!",width=200)
        submit = st.button("Predict")
    with col2:
        st.markdown("""
                    Did you know that machine learning models can help you
        predict heart disease pretty accurately? In this app, you can
        estimate your chance of heart disease (yes/no) in seconds!
        
        Here, a Random Forest model is used with an oversmapling technique
        was constructed using survey data of over 300k US residents.
        This application is based on it because it has proven to be better than the random forest
        (it achieves an accuracy of about 96%, which is quite better then other algo of machine learning).
        
        To predict your heart disease status, simply follow the steps bellow:
        1. Enter the parameters that best describe you;
        2. Press the "Predict" button and wait for the result.
            
        **Keep in mind that this results is not equivalent to a medical diagnosis!
        This model would never be adopted by health care facilities because of its less
        than perfect accuracy, so if you have any problems, consult a human doctor.** """)
    
    heart = loadDataset()
    st.sidebar.title("Feature Selection:")
    userInput = userInterFace()
    
    if submit:
        prediction = loaded_model.predict(userInput)
        if prediction == 0:
            st.warning("Your Heart Is Healthy !! ")
        else:
            st.warning("Your Heart Not Is Healthy !! ")
            
    
    
if __name__ == "__main__":
    main()
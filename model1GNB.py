# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 23:52:15 2021

@author: asus
"""
import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_health_model_nb.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()
disease = list(data.keys())
for i in disease:
    temp = i
    globals()[temp]=i

dead = disease[2:]
classifier = data["model"]
lab_encoder = data["encoder"]
df = data["dataset"]
df.columns

def predicted_page():
    st.title("Disease Prediction")
    st.write("""### We are performing classification based on 5 symptoms only """)
    all_disease = tuple(dead)
    n = len(all_disease)
    all_symptom = []
    
    symptom1 = st.selectbox("Symtom 1", all_disease)
    all_symptom.append(symptom1)
    
    symptom2 = st.selectbox("Symtom 2", all_disease)
    all_symptom.append(symptom2)
    
    symptom3 = st.selectbox("Symtom 3", all_disease)
    all_symptom.append(symptom3)
    
    symptom4 = st.selectbox("Symtom 4", all_disease)
    all_symptom.append(symptom4)
    
    symptom5 = st.selectbox("Symtom 5", all_disease)
    all_symptom.append(symptom5)
    
    ok = st.button("""### Predict Disease """)
    if ok:
        inp = np.zeros(n)
        for i in all_symptom:
            ind = dead.index(i)
            inp[ind] =1
        
        inp = np.array(inp).reshape(1,-1)
        value = classifier.predict(inp)
        dis = lab_encoder.inverse_transform(value)
        pre = (df.loc[df['disease'] == dis[0]])
        st.subheader("Estimated disease is : ")
        st.subheader(pre)
import streamlit as st
import pickle
import numpy as np
import requests
from streamlit_lottie import st_lottie 

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

url = "https://assets6.lottiefiles.com/private_files/lf30_wo5lnbyz.json"
res_json = load_lottieurl(url)
st_lottie(res_json)




# import the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("Laptop Price Predictor")

# brand
company = st.selectbox('Brand',df['Company'].unique())

# type of laptop
type = st.selectbox('Type',df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the Laptop(Kg)',min_value=0.5)

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.selectbox('IPS',['No','Yes'])

# screen size
screen_size = st.number_input('Screen Size(Inch)',min_value=7)

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = st.selectbox('CPU',df['Cpu'].unique())

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU',df['Gpu'].unique())

os = st.selectbox('OS',df['OpSys'].unique())

if st.button('Predict Price'):
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([company,type,cpu,ram,gpu,os,weight,touchscreen,ips,ppi,ssd])

    query = query.reshape(1,11)
    st.title("The predicted price of this configuration is ₹" + str(int(np.exp(pipe.predict(query)[0]))))
    st.balloons()

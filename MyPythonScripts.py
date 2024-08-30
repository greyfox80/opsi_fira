import streamlit as st
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
#from tensorflow.python.keras.layers import Dense
from function import classify
from PIL import Image
import time

# Title
st.title("Welcome to Shafira and Adenia Contagious Predictions")

# Header
st.header("in here you can see which contagious garbage for your environtment")
st.markdown(" ###### upload you garbage image in uploder below and we will predict which level contagious the garbage ")


# Upload the File
file = st.file_uploader("Upload you garbage image in uploder below and we will predict which level contagious the garbage", type=['jpeg','jpg','png'])

# Load Model Classification
#model = pickle.load(open('D:/Model_Opsi_Data_Sampah_002.sav', 'rb'))
model  = load_model('Model_Opsi_Data_Sampah_003.keras')


# Load_ Class Name
class_names = ['1', '2', '3']
class_name2 = ['Low','Medium','High']
class_name3 = ['Siaga, hubungi rt/rw setempat', 'Waspada, segera hubungi rt/rw dan rencanakan pembersihan','Berbahaya, langsung koordinasikan dengan semua pihak dan segera ambil tindakan pembersihan']




# Display Image
if file is not None:

    image = Image.open(file)
    st.image(image, use_column_width=True)
    
    # Classify image
    class_name, index, prob = classify(image, model, class_names)

    progress_bar = st.progress(0)
    for perc_completed in range(100):
        if (perc_completed == 10):
            st.markdown(" ###### Analyzing & Processing")
        time.sleep(0.01)        
        progress_bar.progress(perc_completed+1)
    st.markdown(" ###### Process Completed !")

    with st.expander('Click for prediction Result :'):
         st.write("## Prediction Level : {} ".format(class_name2[index]))
         st.write("## Prediction Prob : {} ".format(prob))
         st.write("## Action Recommendation : {} ".format(class_name3[index]))






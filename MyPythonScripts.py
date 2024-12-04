import streamlit as st
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
#from tensorflow.python.keras.layers import Dense
from function import classify
from PIL import Image
import time
from annotated_text import annotated_text

# Title
st.title("Welcome to 'EcoEye' Predictions")

# Header
st.header("Classify Contamination in your Environment")

# Upload the File
file = st.file_uploader("Upload you garbage image in uploder below and we will predict which level contagious the garbage", type=['jpeg','jpg','png'])

col1, col2, col3 = st.columns(3)    
with col1:
    """
    """
with col2:
    """
    """    
with col3:
    annotated_text(
                ("by","Shafira","#8ef"),("and","Adenia","#faa")
              )

# Load Model Classification
model  = load_model('Model_Opsi_Data_Sampah_003.keras')


# Load_ Class Name
class_names = ['1', '2', '3']
class_name2 = ['Low','Medium','High']
class_name3 = ['Siaga, Lingkungan perlu di monitor oleh warga', 'Waspada, Warga di sarankan segera menjadwalkan pembersihan lingkungan','Berbahaya, Segera lakukan pembersihan linkungan']


# Display Image
if file is not None:

    image = Image.open(file)
    st.image(image, use_column_width=True)
    progress_bar = st.progress(0)
    for perc_completed in range(100):
        if (perc_completed == 10):
            st.markdown(" ###### Analyzing & Processing")
        time.sleep(0.06)        
        progress_bar.progress(perc_completed+1)

    # Classify image
    class_name, index, prob = classify(image, model, class_names)

    st.markdown(" ###### Process Completed !")

    #with st.expander('Click for prediction Result :'):
         #metrics = st.metric(label="Prediction", value=format(class_name2[index]), delta=format(prob))
         st.write("## Prediction Level : {} ".format(class_name2[index]))
         st.write("## Prediction Prob : {:.0%} ".format(prob, '.0%'))
         st.write("## Action Recommendation : {} ".format(class_name3[index]))






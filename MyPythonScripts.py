import streamlit as st
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
#from tensorflow.python.keras.layers import Dense
from function import classify
from PIL import Image
import time
from annotated_text import annotated_text

# Function to check Keras version compatibility (optional)
def check_keras_version(saved_model_path):
    # Implement logic to check the saved model's Keras version (e.g., from metadata)
    # If versions differ, raise a warning or suggest installing a compatible version
# Load Model Classification
    try:
        model = load_model('Model_Opsi_Data_Sampah_003.keras')
        # Optional: Check Keras version compatibility using check_keras_version()
    except OSError as e:
        st.error(f"Error loading model: {e}")
        exit(1)  # Exit gracefully
    # Load Class Names (assuming pickle format)
    try:
        with open('class_names.pkl', 'rb') as f:  # Replace with your class name file path
            class_names = pickle.load(f)
    except FileNotFoundError:
        st.error("Class names file not found.")
    exit(1)

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
#model = pickle.load(open('D:/Model_Opsi_Data_Sampah_002.sav', 'rb'))
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

    with st.expander('Click for prediction Result :'):
         #metrics = st.metric(label="Prediction", value=format(class_name2[index]), delta=format(prob))
         st.write("## Prediction Level : {} ".format(class_name2[index]))
         st.write("## Prediction Prob : {:.0%} ".format(prob, '.0%'))
         st.write("## Action Recommendation : {} ".format(class_name3[index]))






import streamlit as st
import os
import tensorflow as tf
import numpy as np

st.title('Flower Identification')

# Load the model
#model_path = "/content/drive/MyDrive/Web App/saved_model"
model_path = "C:\\Users\\mdbar\\Desktop\\flower identification web app\\saved_model"
model = tf.keras.models.load_model("model_path") 

data_cat = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
img_height = 224
img_width = 224

# Default folder path
#default_folder_path = "/content/drive/MyDrive/Web App/identity_fish/"
default_folder_path = "C:\\Users\\mdbar\\Desktop\\flower identification web app\\identity_flower"

# Get user input for the uploaded image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file
    file_name = os.path.join(default_folder_path, uploaded_file.name)
    with open(file_name, "wb") as f:
        f.write(uploaded_file.read())

    try:
        # Load and preprocess the uploaded image
        image_load = tf.keras.utils.load_img(file_name, target_size=(img_height, img_width))
        img_arr = tf.keras.utils.img_to_array(image_load)
        img_bat = tf.expand_dims(img_arr, 0)

        # Make predictions
        predict = model.predict(img_bat)
        score = tf.nn.softmax(predict)

        # Display the image
        st.image(image_load, width=400)

        # Display the prediction results
        st.write('Flower in image is ' + data_cat[np.argmax(score)])
        st.write('With Accuracy of ' + str(np.max(score) * 100))
    except Exception as e:
        st.write(f"Error: {e}")
# python -m streamlit run app.py 
import tensorflow as tf
import numpy as np
from PIL import Image
import streamlit as st


model = tf.keras.models.load_model('./models/best_vgg19.h5')

def preprocess_image(image):
    # img = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
    image = Image.open(image)
    image = image.resize((224, 224))
    # Convert the image to an array
    img_array = tf.keras.preprocessing.image.img_to_array(image) 
    img_array = np.expand_dims(img_array, axis=0)  # Expands the dimensions of the image 
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array) 
    return img_array


def make_prediction(model, image):
    class_names = ['ants', 'bees', 'beetle', 'catterpillar', 'earthworms', 'earwig', 'grasshopper', 'moth', 'slug', 'snail', 'wasp', 'weevil']
    predictions = model.predict(image)

    # Decode the prediction from the model
    predicted_class = np.argmax(predictions, axis=-1)  # Highest probability value from predictions
    prediction = class_names[predicted_class[0]] 

    return prediction


# Streamlit
st.title('Pest Detection Model')

upload = st.file_uploader('Upload an image to predict:', type='jpg')

if upload:
    img = Image.open(upload)
    st.image(img, caption='Uploaded image', use_column_width=True)

    image = preprocess_image(upload)
    prediction = make_prediction(model, image)

    st.write('Prediction:', prediction.title())
    st.write('Now spraying pesticide on detected pest!')

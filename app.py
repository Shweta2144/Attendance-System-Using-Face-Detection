import streamlit as st
import numpy as np
import sqlite3
import cv2
from deepface import DeepFace
from PIL import Image
import io

# Load database
conn = sqlite3.connect("face_db.sqlite")
c = conn.cursor()

# Function to extract vector
def extract_face_vector(image):
    try:
        embedding = DeepFace.represent(img_path=image, model_name="Facenet")[0]["embedding"]
        return np.array(embedding).astype(np.float32)
    except:
        return None

# Function to compare vectors
def compare_faces(uploaded_vector, stored_vector):
    return np.linalg.norm(uploaded_vector - stored_vector) < 10  # Set similarity threshold

# Streamlit UI
st.title("Face Recognition System")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to vector
    img_array = np.array(image)
    uploaded_vector = extract_face_vector(img_array)

    if uploaded_vector is not None:
        c.execute("SELECT name,enrollement,contact, dob, address, branch, first_year_p,second_year_p,third_year_p,face_vector FROM faces")
        faces = c.fetchall()

        match_found = False
        for name,enrollement,contact, dob, address, branch, first_year_p,second_year_p,third_year_p,face_blob in faces:
            stored_vector = np.frombuffer(face_blob, dtype=np.float32)
            if compare_faces(uploaded_vector, stored_vector):
                st.success(f"Match Found! \n\n**Name:** {name}\n**Enrollement:** {enrollement}\n**Contact:** {contact}\n**Dob:** {dob}\n**Address:** {address}\n**Branch:** {branch}\n**First_year_p:** {first_year_p}\n**Second_year_p:** {second_year_p}\n**Third_year_p:** {third_year_p}")
                match_found = True
                break

        if not match_found:
            st.error("No Match Found in Database")
    else:
        st.error("Face not detected in the uploaded image")

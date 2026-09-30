import streamlit as st
import sqlite3
import numpy as np
import cv2
import tempfile
import os
from deepface import DeepFace
from PIL import Image
import sys

# Connect to SQLite database
conn = sqlite3.connect("face_db2.sqlite", check_same_thread=False)
c = conn.cursor()

# Create table if not exists
c.execute("""
    CREATE TABLE IF NOT EXISTS faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
          branch TEXT,
          enrollment INTEGER,
        age INTEGER,
        address TEXT,
        mobile TEXT,
        face_vector BLOB
    )
""")
conn.commit()

# Function to extract face embedding using DeepFace
def extract_face_vector(image_path):
    try:
        embedding = DeepFace.represent(img_path=image_path, model_name="Facenet")[0]["embedding"]
        return np.array(embedding).astype(np.float32)
    except:
        return None

# Function to add face to database
def add_face(name,branch,enrollment, age, address, mobile, image_path):
    face_vector = extract_face_vector(image_path)
    if face_vector is not None:
        c.execute("INSERT INTO faces (name,branch,enrollment, age, address, mobile, face_vector) VALUES (?,?,?, ?, ?, ?, ?)", 
                  (name,branch,enrollment, age, address, mobile, face_vector.tobytes()))
        conn.commit()
        st.success(f"✅ {name} added successfully!")
    else:
        st.error("⚠️ No face detected in the uploaded image.")

# Streamlit UI
st.title("Add New Face to Database")

# User Inputs
name = st.text_input("Enter Name")
age = st.number_input("Enter Age", min_value=1, max_value=120)
address = st.text_area("Enter Address")
mobile = st.text_input("Enter Mobile Number")
branch= st.text_input("Enter Branch")
enrollment= st.text_input("Enter Enrollment")

# Upload Image
uploaded_file = st.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save image temporarily to pass file path to DeepFace
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file_path = temp_file.name
        image.save(temp_file_path)  # Save uploaded image to a temporary file

    if st.button("Save to Database"):
        add_face(name, branch,enrollment,age, address, mobile, temp_file_path)
        os.remove(temp_file_path)  # Remove temp file after use
if st.button("Close App"):
    st.warning("Closing the app...")
    sys.exit(0)  # Exits the script
import streamlit as st
import sqlite3
import numpy as np
import cv2
import tempfile
import os
from deepface import DeepFace
from PIL import Image
import sys


conn = sqlite3.connect("face_db.sqlite", check_same_thread=False)
c = conn.cursor()


c.execute("""
    CREATE TABLE IF NOT EXISTS faces (
        name TEXT,
        enrollement INTEGER PRIMARY KEY,
        contact TEXT,
        dob DATE,
        address TEXT,
        branch TEXT,
        first_year_p TEXT,
        second_year_p TEXT,
        third_year_p TEXT,
        face_vector BLOB
    )
""")
conn.commit()

def extract_face_vector(image_path):
    try:
        embedding = DeepFace.represent(img_path=image_path, model_name="Facenet")[0]["embedding"]
        return np.array(embedding).astype(np.float32)
    except:
        return None

def add_face(name,enrollement,contact, dob ,address, branch,first_year_p,second_year_p,third_year_p, image_path):
    face_vector = extract_face_vector(image_path)
    if face_vector is not None:
        c.execute("INSERT INTO faces (name,enrollement, contact,dob, address,branch, first_year_p,second_year_p,third_year_p, face_vector) VALUES (?, ?, ?, ?, ?,?,?,?,?,?)", 
                  (name,enrollement, contact, dob,address, branch,first_year_p, second_year_p,third_year_p,face_vector.tobytes()))
        conn.commit()
        st.success(f"✅ {name} added successfully!")
    else:
        st.error("⚠️ No face detected in the uploaded image.")


st.title("Add New Face to Database")


name = st.text_input("Enter Name")
enrollement=st.text_input("ENTER ENROLLEMENT")
contact=st.text_input("ENTER CONTACT")
dob=st.date_input("ENTER DTAE OF BIRTH")
address = st.text_area("Enter Address")
branch=st.text_input("ENTER BRANCH")
first_year_p=st.text_input("ENTER FIRST YEAR PERCENTAGE")
second_year_p=st.text_input("ENTER SECOND YEAR PERCETAGE")
third_year_p=st.text_input("ENTER THIRD YEAR PERCENTAGE")



uploaded_file = st.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file_path = temp_file.name
        image.save(temp_file_path)  

    if st.button("Save to Database"):
        add_face(name,enrollement,contact,dob, address, branch,first_year_p, second_year_p,third_year_p,temp_file_path)
        os.remove(temp_file_path)  
if st.button("Close App"):
    st.warning("Closing the app...")
    sys.exit(0)  
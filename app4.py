import streamlit as st
import numpy as np
import sqlite3
from deepface import DeepFace
from PIL import Image
import io
import datetime


conn_faces = sqlite3.connect("face_db.sqlite")
conn_attendance = sqlite3.connect("attendance_db.sqlite")
c_faces = conn_faces.cursor()
c_attendance = conn_attendance.cursor()


def extract_face_vector(image):
    try:
        embedding = DeepFace.represent(img_path=image, model_name="Facenet")[0]["embedding"]
        return np.array(embedding).astype(np.float32)
    except:
        return None


def compare_faces(uploaded_vector, stored_vector):
    return np.linalg.norm(uploaded_vector - stored_vector) < 10  

def record_attendance(name):
    date_today = datetime.date.today().strftime("%Y-%m-%d")
    c_attendance.execute("CREATE TABLE IF NOT EXISTS attendance (name TEXT, date TEXT, UNIQUE(name, date))")
    c_attendance.execute("SELECT * FROM attendance WHERE name=? AND date=?", (name, date_today))
    if c_attendance.fetchone() is None:
        c_attendance.execute("INSERT INTO attendance (name, date) VALUES (?, ?)", (name, date_today))
        conn_attendance.commit()
        st.success(f"Attendance recorded for {name} on {date_today}")
    else:
        st.info(f"Attendance already recorded for {name} today.")


st.title("Face Recognition System")


captured_image = st.camera_input("Take a photo")

if captured_image is not None:
 
    image = Image.open(captured_image)
    st.image(image, caption="Captured Image", use_column_width=True)

    
    img_array = np.array(image)

    
    uploaded_vector = extract_face_vector(img_array)

    if uploaded_vector is not None:
        c_faces.execute("SELECT name,enrollement,contact,dob,address, branch,first_year_p,second_year_p,third_year_p,face_vector FROM faces")
        faces = c_faces.fetchall()

        match_found = False
        for name,enrollment, contact,dob,address,branch,first_year_p , second_year_p,third_year_p,face_blob in faces:
            stored_vector = np.frombuffer(face_blob, dtype=np.float32)
            if compare_faces(uploaded_vector, stored_vector):
                st.success(f"Match Found! \n\n**Name:** {name}\n\n**Enrollement:** {enrollment}\n\n**Contact:** {contact}\n\n**Dob:** {dob}\n\n**Address:** {address}\n\n**Branch:** {branch}\n\n**First_year_p:** {first_year_p}\n\n**Second_year_p:** {second_year_p}\n\n**Third_year_p:** {third_year_p}")
                record_attendance(name)
                match_found = True
                break

        if not match_found:
            st.error("No Match Found in Database")
    else:
        st.error("Face not detected in the captured image")

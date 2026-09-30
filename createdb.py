import os
import cv2
import numpy as np
import sqlite3
from deepface import DeepFace


conn = sqlite3.connect("face_db5.sqlite")
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


def add_face(name, enrollement,contact,dob, address,branch,first_year_p, second_year_p,third_year_p,image_path):
    face_vector = extract_face_vector(image_path)
    if face_vector is not None:
        c.execute("INSERT INTO faces (name, enrollement, contact,dob,address, branch,first_year_p,second_year_p,third_year_p, face_vector) VALUES (?, ?, ?, ?, ?)", 
                  (name,enrollement,contact,dob, address, branch ,first_year_p,second_year_p,third_year_p, face_vector.tobytes()))
        conn.commit()
        print(f"Added {name} to database")
    else:
        print("Face not detected")

conn.close()

import os
import cv2
import numpy as np
import sqlite3
from deepface import DeepFace


conn = sqlite3.connect("face_db6.sqlite")
c = conn.cursor()
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


def extract_face_vector(image_path):
    try:
        embedding = DeepFace.represent(img_path=image_path, model_name="Facenet")[0]["embedding"]
        return np.array(embedding).astype(np.float32)
    except:
        return None


def add_face(name,branch,enrollment, age, address, mobile, image_path):
    face_vector = extract_face_vector(image_path)
    if face_vector is not None:
        c.execute("INSERT INTO faces (name,branch,enrollment,age, address, mobile, face_vector) VALUES (?,?,?,?,?, ?, ?)", 
                  (name,branch,enrollment, age, address, mobile, face_vector.tobytes()))
        conn.commit()
        print(f"Added {name} to database")
    else:
        print("Face not detected")

conn.close()

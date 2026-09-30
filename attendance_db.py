import os
import cv2
import numpy as np
import sqlite3
conn = sqlite3.connect("attendance_db.sqlite")
c = conn.cursor()
c.execute("""
    CREATE TABLE IF NOT EXISTS attendance (name TEXT, date TEXT)
""")
conn.commit()
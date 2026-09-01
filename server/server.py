"""
Headless TCP server for face recognition (UfaHack2024).
Reuses the recognition logic from the desktop app but runs standalone in Docker.
No GUI dependencies (customtkinter / Tkinter).
"""

import os
import socket
import tempfile
import pickle
import logging
import sys

import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
from mtcnn import MTCNN
from catboost import CatBoostClassifier

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "12345"))
MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")
DATA_DIR = os.environ.get("DATA_DIR", "/app/data")

# ---------------------------------------------------------------------------
# Logging to stdout so Docker logs capture everything
# ---------------------------------------------------------------------------
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("server")

# ---------------------------------------------------------------------------
# Globals (loaded once at startup)
# ---------------------------------------------------------------------------
detector = None
catboost_model = None
name_dict = None


def load_models():
    """Load MTCNN detector, CatBoost model and name dictionary.
    Logs warnings instead of crashing when files are absent so the container
    starts and waits for connections even without .cbm files."""
    global detector, catboost_model, name_dict

    detector = MTCNN()
    log.info("MTCNN detector initialised")

    model_path = os.path.join(MODEL_DIR, "catboost_usa.cbm")
    if os.path.isfile(model_path):
        try:
            catboost_model = CatBoostClassifier()
            catboost_model.load_model(model_path)
            log.info("CatBoost model loaded from %s", model_path)
        except Exception as exc:
            log.warning("Failed to load CatBoost model from %s: %s — recognition will be unavailable", model_path, exc)
            catboost_model = None
    else:
        log.warning("CatBoost model not found at %s — recognition will be unavailable until it is provided", model_path)

    dict_path = os.path.join(MODEL_DIR, "saved_dictionary.pkl")
    if os.path.isfile(dict_path):
        try:
            with open(dict_path, "rb") as f:
                name_dict = pickle.load(f)
            log.info("Name dictionary loaded from %s", dict_path)
        except Exception as exc:
            log.warning("Failed to load name dictionary from %s: %s", dict_path, exc)
            name_dict = None
    else:
        log.warning("Name dictionary not found at %s — recognition will be unavailable until it is provided", dict_path)


def recognise_face(image_array):
    """Run the full recognition pipeline on an RGB image array.
    Returns (name, confidence_info) or an error message string."""
    if catboost_model is None or name_dict is None:
        return "recognition unavailable (models not loaded)"

    detections = detector.detect_faces(image_array)
    if not detections:
        return "no faces detected"

    # Use the first face with confidence > 0.9
    for detection in detections:
        confidence = detection["confidence"]
        if confidence > 0.9:
            x, y, w, h = detection["box"]
            detected_face = image_array[int(y) : int(y + h), int(x) : int(x + w)]

            embedding = DeepFace.represent(
                detected_face, model_name="Facenet", enforce_detection=False
            )
            ebd = embedding[0]["embedding"]

            # Build DataFrame exactly as Predict_photo.py does
            dicter3 = {1: ebd}
            data_usa = pd.DataFrame.from_dict(dicter3.items())
            data_usa.rename(columns={0: "id", 1: "embd"}, inplace=True, errors="ignore")
            new_cols = pd.DataFrame(data_usa["embd"].apply(pd.Series))
            df_usa = pd.concat([data_usa, new_cols], axis=1)
            df_usa.drop(["embd"], axis=1, inplace=True, errors="ignore")
            X = df_usa.drop(["id"], axis=1)

            result = catboost_model.predict(X)
            idx = result[0][0]
            name = name_dict.get(idx, f"unknown (index {idx})")
            log.info("Recognised: %s (confidence %.3f)", name, confidence)
            return name

    return "no face with sufficient confidence (>0.9)"


def handle_client(conn, addr):
    """Receive a photo from one client, recognise the face and send the name back."""
    log.info("Accepted connection from %s", addr)

    # Read the full photo bytes until the connection closes
    chunks = []
    while True:
        chunk = conn.recv(65536)
        if not chunk:
            break
        chunks.append(chunk)
    photo_bytes = b"".join(chunks)

    if not photo_bytes:
        log.info("Empty request from %s — closing", addr)
        conn.close()
        return

    log.info("Received %d bytes from %s", len(photo_bytes), addr)

    # Write photo to OS temp dir (not FDJ.jpg in CWD)
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    try:
        tmp.write(photo_bytes)
        tmp.close()

        img = cv2.imread(tmp.name)
        if img is None:
            result = "failed to decode image"
            log.warning("Could not decode image from %s", addr)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (1080, 720))
            result = recognise_face(img)

        # Send plain UTF-8 bytes (NOT bit-string encoding — bug fixed)
        conn.sendall(result.encode("utf-8"))
    finally:
        os.unlink(tmp.name)

    conn.close()
    log.info("Closed connection from %s", addr)


def main():
    log.info("Starting UfaHack2024 headless server")
    log.info("HOST=%s, PORT=%s, MODEL_DIR=%s, DATA_DIR=%s", HOST, PORT, MODEL_DIR, DATA_DIR)

    load_models()

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((HOST, PORT))
    server_sock.listen(5)
    log.info("Listening on %s:%s", HOST, PORT)

    while True:
        conn, addr = server_sock.accept()
        try:
            handle_client(conn, addr)
        except Exception as exc:
            log.exception("Error handling client %s: %s", addr, exc)


if __name__ == "__main__":
    main()
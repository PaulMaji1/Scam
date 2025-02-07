import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import tempfile

# 🚀 Cargar el modelo YOLOv8 (ajusta la ruta si es necesario)
@st.cache_resource
def load_model():
    model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt", force_reload=True)
    return model

model = load_model()

# 🎨 Interfaz de Streamlit
st.title("Segmentación de Imágenes con YOLO 🚀")
st.write("Sube una imagen y el modelo realizará la segmentación.")

# 📤 Subida de imagen
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 📸 Mostrar imagen original
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen Cargada", use_column_width=True)

    # Convertir a formato numpy
    img = np.array(image)

    # 🧠 Realizar la segmentación
    results = model(img)

    # 📌 Dibujar la segmentación en la imagen
    segmented_img = np.squeeze(results.render())

    # Mostrar la imagen segmentada
    st.image(segmented_img, caption="Segmentación Completa", use_column_width=True)

    # 📥 Descargar imagen segmentada
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        cv2.imwrite(temp_file.name, cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR))
        st.download_button("Descargar Imagen Segmentada", temp_file.name, file_name="segmentacion.jpg")


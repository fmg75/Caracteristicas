import os
import io
import streamlit as st
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import pickle
import uuid
import base64
import tempfile

# --- Configuración inicial de la página ---
st.set_page_config(
    page_title="DeepSeek Chat",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Toggle de tema Claro/Oscuro en la barra lateral ---
theme = st.sidebar.radio("Seleccionar tema:", ("Claro", "Oscuro"), index=0)
if theme == "Oscuro":
    st.markdown(
        """
        <style>
        /* Fondo general */
        .css-1d391kg {background-color: #0e1117 !important;}
        /* Texto */
        body, .css-1d391kg, .css-jn99sy p {color: #fafafa !important;}
        /* Entradas y botones */
        .stButton>button, .stFileUploader>div {background-color: #1e1e1e !important; color: #fafafa !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        /* Restablecer tema claro nativo */
        .css-1d391kg {background-color: #ffffff !important;}
        body, .css-jn99sy p {color: #000000 !important;}
        .stButton>button, .stFileUploader>div {background-color: #f0f2f6 !important; color: #000000 !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )

# --- Generador de ID único para descargas ---
unique_id = str(uuid.uuid4())[:8]

class FaceNetModels:
    def __init__(self):
        # Cargar modelo preentrenado y detector MTCNN
        self.model = InceptionResnetV1(pretrained="vggface2").eval()
        self.mtcnn = MTCNN(min_face_size=50, keep_all=False)
        self.caracteristicas = None

    def load_caracteristicas(self, filename):
        with open(filename, "rb") as f:
            self.caracteristicas = pickle.load(f)

    def embedding(self, img_tensor):
        return self.model(img_tensor.unsqueeze(0))

    def Distancia(self, img_embedding):
        distances = [
            (label, torch.dist(emb, img_embedding))
            for label, emb in self.caracteristicas.items()
        ]
        label, dist = min(distances, key=lambda x: x[1])
        return label, dist.item()

    def extract_embeddings(self, uploaded_files):
        embeddings_list, labels, no_process = [], [], []
        for f in uploaded_files:
            img = Image.open(f).convert("RGB")
            face = self.mtcnn(img)
            if face is None:
                no_process.append(f.name)
                continue
            embeddings_list.append(self.model(face.unsqueeze(0)))
            labels.append(os.path.splitext(f.name)[0])
        self.caracteristicas = dict(zip(labels, embeddings_list))
        st.success(f"Procesadas: {len(embeddings_list)} imágenes.")
        if no_process:
            st.warning(f"No procesadas: {', '.join(no_process)}")
        return self.caracteristicas

# --- Función para extracción de características ---
def feature_extraction(uploaded_files):
    models = FaceNetModels()
    if st.button("Extraer características"):
        caracteristicas = models.extract_embeddings(uploaded_files)
        fname = f"features_{unique_id}.pkl"
        with open(fname, "wb") as out:
            pickle.dump(caracteristicas, out)
        with open(fname, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="{fname}">Descargar .pkl</a>'
        st.markdown(href, unsafe_allow_html=True)

# --- Función para reconocimiento facial ---
def upload_and_process_image(uploaded_file, pkl_file):
    try:
        models = FaceNetModels()
        # Guardar .pkl temporalmente
        tmp = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
        tmp.write(pkl_file.read())
        tmp.close()
        models.load_caracteristicas(tmp.name)
        img = Image.open(io.BytesIO(uploaded_file.read()))
        if img.format == "PNG":
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG")
            buf.seek(0)
            img = Image.open(buf)
        face_tensor = models.mtcnn(img)
        if face_tensor is None:
            st.error("No se detectó rostro en la imagen proporcionada.")
            return
        emb = models.embedding(face_tensor)
        label, dist = models.Distancia(emb)
        st.image(img, width=200)
        similarity = max(0, int(100 - 17.14 * dist))
        st.write(f"Posible: **{label}** (Similitud: {similarity}%)")
    except Exception as e:
        st.error(f"Error: {e}")

# --- Barra lateral e interfaz principal ---
st.sidebar.title("Opciones")
mode = st.sidebar.selectbox("Menú:", ["Generar características", "Reconocer rostro"])

if mode == "Generar características":
    files = st.file_uploader("Subir imágenes (JPEG/PNG)", type=['jpg','jpeg','png'], accept_multiple_files=True)
    if files:
        feature_extraction(files)
else:
    uploaded = st.file_uploader("Subir imagen a reconocer", type=['jpg','jpeg','png'])
    pkl = st.file_uploader("Subir diccionario .pkl", type=['pkl'])
    if uploaded and pkl:
        upload_and_process_image(uploaded, pkl)

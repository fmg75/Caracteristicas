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
import numpy as np

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
        # Configurar dispositivo
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cargar modelo preentrenado y detector MTCNN
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.mtcnn = MTCNN(
            min_face_size=50, 
            keep_all=False, 
            device=self.device,
            post_process=False  # Evitar problemas de conversión automática
        )
        self.caracteristicas = None

    def load_caracteristicas(self, filename):
        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)
                
            # Convertir datos si es necesario
            self.caracteristicas = {}
            for label, embedding in data.items():
                if isinstance(embedding, np.ndarray):
                    # Convertir numpy array a tensor
                    embedding = torch.from_numpy(embedding).float().to(self.device)
                elif isinstance(embedding, torch.Tensor):
                    # Asegurar que el tensor esté en el dispositivo correcto
                    embedding = embedding.float().to(self.device)
                
                self.caracteristicas[label] = embedding
                
        except Exception as e:
            st.error(f"Error al cargar características: {str(e)}")
            raise e

    def embedding(self, img_tensor):
        """Extrae embedding de un tensor de imagen"""
        with torch.no_grad():
            img_tensor = img_tensor.to(self.device)
            if len(img_tensor.shape) == 3:
                img_tensor = img_tensor.unsqueeze(0)
            return self.model(img_tensor)

    def Distancia(self, img_embedding):
        """Calcula distancia con embeddings almacenados"""
        if self.caracteristicas is None:
            raise ValueError("No se han cargado características")
            
        img_embedding = img_embedding.to(self.device)
        distances = []
        
        for label, emb in self.caracteristicas.items():
            emb = emb.to(self.device)
            # Asegurar que ambos tensores tengan la misma forma
            if len(img_embedding.shape) != len(emb.shape):
                if len(img_embedding.shape) == 2 and img_embedding.shape[0] == 1:
                    img_embedding = img_embedding.squeeze(0)
                if len(emb.shape) == 2 and emb.shape[0] == 1:
                    emb = emb.squeeze(0)
            
            dist = torch.dist(emb, img_embedding)
            distances.append((label, dist))
            
        label, dist = min(distances, key=lambda x: x[1])
        return label, dist.item()

    def extract_embeddings(self, uploaded_files):
        """Extrae embeddings de múltiples archivos"""
        embeddings_list, labels, no_process = [], [], []
        
        for f in uploaded_files:
            try:
                # Cargar imagen
                img = Image.open(f).convert("RGB")
                
                # Detectar rostro
                face = self.mtcnn(img)
                
                if face is None:
                    no_process.append(f.name)
                    continue
                
                # Extraer embedding
                with torch.no_grad():
                    face = face.to(self.device)
                    embedding = self.model(face.unsqueeze(0))
                    
                embeddings_list.append(embedding.cpu())  # Guardar en CPU para pickle
                labels.append(os.path.splitext(f.name)[0])
                
            except Exception as e:
                st.error(f"Error procesando {f.name}: {str(e)}")
                no_process.append(f.name)
                continue
        
        # Crear diccionario de características
        self.caracteristicas = dict(zip(labels, embeddings_list))
        
        st.success(f"Procesadas: {len(embeddings_list)} imágenes.")
        if no_process:
            st.warning(f"No procesadas: {', '.join(no_process)}")
            
        return self.caracteristicas

# --- Función para extracción de características ---
def feature_extraction(uploaded_files):
    models = FaceNetModels()
    
    if st.button("Extraer características"):
        with st.spinner("Procesando imágenes..."):
            try:
                caracteristicas = models.extract_embeddings(uploaded_files)
                
                if not caracteristicas:
                    st.error("No se pudieron procesar imágenes.")
                    return
                
                # Guardar archivo pickle
                fname = f"features_{unique_id}.pkl"
                with open(fname, "wb") as out:
                    pickle.dump(caracteristicas, out, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Crear enlace de descarga
                with open(fname, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{fname}">📥 Descargar archivo .pkl</a>'
                
                st.markdown(href, unsafe_allow_html=True)
                st.success(f"Archivo {fname} generado correctamente!")
                
                # Limpiar archivo temporal
                try:
                    os.remove(fname)
                except:
                    pass
                    
            except Exception as e:
                st.error(f"Error en la extracción: {str(e)}")

# --- Función para reconocimiento facial ---
def upload_and_process_image(uploaded_file, pkl_file):
    try:
        models = FaceNetModels()
        
        # Guardar .pkl temporalmente
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp.write(pkl_file.read())
            tmp_path = tmp.name
        
        # Cargar características
        models.load_caracteristicas(tmp_path)
        
        # Procesar imagen subida
        uploaded_file.seek(0)  # Resetear puntero del archivo
        img = Image.open(uploaded_file).convert("RGB")
        
        # Detectar rostro
        with st.spinner("Detectando rostro..."):
            face_tensor = models.mtcnn(img)
            
        if face_tensor is None:
            st.error("❌ No se detectó rostro en la imagen proporcionada.")
            return
        
        # Extraer embedding y comparar
        with st.spinner("Comparando con base de datos..."):
            emb = models.embedding(face_tensor)
            label, dist = models.Distancia(emb)
        
        # Mostrar resultados
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(img, caption="Imagen analizada", width=200)
        
        with col2:
            similarity = max(0, min(100, int(100 - 17.14 * dist)))
            
            if similarity > 70:
                st.success(f"✅ **Reconocido como: {label}**")
                st.metric("Similitud", f"{similarity}%", delta=None)
            elif similarity > 40:
                st.warning(f"⚠️ **Posible match: {label}**")
                st.metric("Similitud", f"{similarity}%", delta=None)
            else:
                st.error(f"❌ **No reconocido** (Mejor match: {label})")
                st.metric("Similitud", f"{similarity}%", delta=None)
        
        # Limpiar archivo temporal
        try:
            os.unlink(tmp_path)
        except:
            pass
            
    except Exception as e:
        st.error(f"❌ Error en el procesamiento: {str(e)}")
        
        # Información de debug (opcional)
        if st.checkbox("Mostrar información de debug"):
            st.write(f"Tipo de error: {type(e).__name__}")
            st.write(f"Detalles: {str(e)}")

# --- Barra lateral e interfaz principal ---
st.sidebar.title("🔍 Opciones")
mode = st.sidebar.selectbox("Menú principal:", ["Generar características", "Reconocer rostro"])

# Información del sistema
with st.sidebar.expander("ℹ️ Información del sistema"):
    st.write(f"PyTorch: {torch.__version__}")
    st.write(f"CUDA disponible: {'✅' if torch.cuda.is_available() else '❌'}")
    st.write(f"Dispositivo: {'GPU' if torch.cuda.is_available() else 'CPU'}")

st.title("🧠 Sistema de Reconocimiento Facial")

if mode == "Generar características":
    st.header("📤 Generar base de datos de características")
    st.write("Sube múltiples imágenes para crear un archivo .pkl con las características faciales.")
    
    files = st.file_uploader(
        "Seleccionar imágenes (JPEG/PNG)", 
        type=['jpg','jpeg','png'], 
        accept_multiple_files=True,
        help="Sube una imagen por persona. El nombre del archivo será usado como etiqueta."
    )
    
    if files:
        st.write(f"📁 {len(files)} archivo(s) seleccionado(s)")
        for f in files:
            st.write(f"- {f.name}")
        feature_extraction(files)

else:
    st.header("🔍 Reconocer rostro")
    st.write("Sube una imagen y un archivo .pkl para identificar la persona.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded = st.file_uploader(
            "📷 Imagen a reconocer", 
            type=['jpg','jpeg','png'],
            help="Imagen con el rostro que quieres identificar"
        )
    
    with col2:
        pkl = st.file_uploader(
            "📄 Archivo de características (.pkl)", 
            type=['pkl'],
            help="Archivo generado en la sección 'Generar características'"
        )
    
    if uploaded and pkl:
        upload_and_process_image(uploaded, pkl)
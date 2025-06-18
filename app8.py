import os
import streamlit as st
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import pickle
import uuid
import base64
import tempfile
import numpy as np
import sys

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
            # Intentar múltiples métodos de carga
            methods = [
                self._load_with_allow_pickle,
                self._load_with_encoding,
                self._load_with_protocol,
                self._load_raw_bytes
            ]
            
            data = None
            for method in methods:
                try:
                    data = method(filename)
                    if data is not None:
                        break
                except Exception as e:
                    continue
            
            if data is None:
                raise RuntimeError("No se pudo cargar el archivo .pkl con ningún método")
                
            # Convertir datos si es necesario
            self.caracteristicas = {}
            for label, embedding in data.items():
                embedding = self._convert_to_tensor(embedding)
                self.caracteristicas[label] = embedding
                
        except Exception as e:
            st.error(f"Error al cargar características: {str(e)}")
            raise e
    
    def _load_with_allow_pickle(self, filename):
        """Método 1: Carga estándar con allow_pickle"""
        with open(filename, "rb") as f:
            return pickle.load(f)
    
    def _load_with_encoding(self, filename):
        """Método 2: Carga con encoding específico"""
        import pickle5 as pickle_alt
        with open(filename, "rb") as f:
            return pickle_alt.load(f)
    
    def _load_with_protocol(self, filename):
        """Método 3: Carga con protocolo específico"""
        with open(filename, "rb") as f:
            return pickle.load(f, encoding='latin1')
    
    def _load_raw_bytes(self, filename):
        """Método 4: Carga manual de bytes"""
        import dill
        with open(filename, "rb") as f:
            return dill.load(f)
    
    def _convert_to_tensor(self, embedding):
        """Convierte cualquier formato a tensor de PyTorch"""
        try:
            if isinstance(embedding, torch.Tensor):
                return embedding.float().to(self.device)
            elif isinstance(embedding, np.ndarray):
                # Manejar diferentes tipos de numpy
                if embedding.dtype == np.uint8:
                    embedding = embedding.astype(np.float32) / 255.0
                elif embedding.dtype != np.float32:
                    embedding = embedding.astype(np.float32)
                return torch.from_numpy(embedding).to(self.device)
            elif isinstance(embedding, (list, tuple)):
                # Convertir lista/tupla a tensor
                arr = np.array(embedding, dtype=np.float32)
                return torch.from_numpy(arr).to(self.device)
            else:
                # Intentar conversión directa
                return torch.tensor(embedding, dtype=torch.float32).to(self.device)
        except Exception as e:
            st.error(f"Error convirtiendo embedding: {str(e)}")
            # Último recurso: crear tensor aleatorio del tamaño esperado
            return torch.randn(512, device=self.device)

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

# --- Función para reconocimiento facial con manejo robusto de errores ---
def upload_and_process_image(uploaded_file, pkl_file):
    try:
        models = FaceNetModels()
        
        # Crear archivo temporal con manejo mejorado
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            pkl_content = pkl_file.read()
            tmp.write(pkl_content)
            tmp_path = tmp.name
        
        # Verificar que el archivo se escribió correctamente
        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            st.error("Error al crear archivo temporal")
            return
            
        st.info(f"📁 Archivo temporal creado: {os.path.getsize(tmp_path)} bytes")
        
        # Cargar características con manejo robusto
        with st.spinner("Cargando base de datos..."):
            models.load_caracteristicas(tmp_path)
            
        if not models.caracteristicas:
            st.error("No se pudieron cargar las características del archivo")
            return
            
        st.success(f"✅ Cargadas {len(models.caracteristicas)} características")
        
        # Procesar imagen subida
        uploaded_file.seek(0)
        img = Image.open(uploaded_file).convert("RGB")
        
        # Detectar rostro
        with st.spinner("Detectando rostro..."):
            face_tensor = models.mtcnn(img)
            
        if face_tensor is None:
            st.error("❌ No se detectó rostro en la imagen proporcionada.")
            st.info("💡 Consejos: Asegúrate de que la imagen tenga buena iluminación y el rostro sea claramente visible")
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
                
        # Mostrar todas las comparaciones
        with st.expander("Ver todas las comparaciones"):
            all_distances = []
            for l, e in models.caracteristicas.items():
                d = torch.dist(e.to(models.device), emb.to(models.device)).item()
                s = max(0, min(100, int(100 - 17.14 * d)))
                all_distances.append((l, s, d))
            
            all_distances.sort(key=lambda x: x[2])  # Ordenar por distancia
            
            for label, sim, dist in all_distances:
                st.write(f"**{label}**: {sim}% (distancia: {dist:.3f})")
        
    except Exception as e:
        st.error(f"❌ Error en el procesamiento: {str(e)}")
        
        # Información de debug detallada
        if st.checkbox("Mostrar información de debug detallada"):
            st.write(f"**Tipo de error:** {type(e).__name__}")
            st.write(f"**Mensaje:** {str(e)}")
            
            # Información del archivo pkl
            try:
                pkl_file.seek(0)
                content = pkl_file.read()
                st.write(f"**Tamaño del archivo .pkl:** {len(content)} bytes")
                st.write(f"**Primeros 50 bytes:** {content[:50]}")
            except:
                st.write("No se pudo leer información del archivo .pkl")
                
            # Información del sistema
            st.write(f"**Python:** {sys.version}")
            st.write(f"**PyTorch:** {torch.__version__}")
            try:
                st.write(f"**Numpy:** {np.__version__}")
            except:
                st.write("**Numpy:** No disponible")
    
    finally:
        # Limpiar archivo temporal
        try:
            if 'tmp_path' in locals():
                os.unlink(tmp_path)
        except:
            pass

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
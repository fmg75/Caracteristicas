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

# Generar un ID único utilizando uuid
unique_id = str(uuid.uuid4())[:8]

class FaceNetModels:
    def __init__(self):
        self.model = InceptionResnetV1(pretrained="vggface2").eval()
        self.mtcnn = MTCNN(min_face_size=50, keep_all=False)
        self.caracteristicas = None

    def load_caracteristicas(self, filename):
        with open(filename, "rb") as f:
            self.caracteristicas = pickle.load(f)

    def embedding(self, img_tensor):
        img_embedding = self.model(img_tensor.unsqueeze(0))
        return img_embedding

    def Distancia(self, img_embedding):
        distances = [
            (label, torch.dist(emb, img_embedding))
            for label, emb in self.caracteristicas.items()
        ]
        sorted_distances = sorted(distances, key=lambda x: x[1])
        return sorted_distances[0][0], sorted_distances[0][1].item()

    def extract_embeddings(self, uploaded_files):
        embeddings_list = []
        labels = []
        no_process_images = []

        for uploaded_file in uploaded_files:
            # Usar bytes directamente en lugar de abrir como archivo
            img = Image.open(uploaded_file)
            img = img.convert("RGB")
            label = os.path.splitext(uploaded_file.name)[0]
            face = self.mtcnn(img)

            if face is None:
                no_process_images.append(uploaded_file.name)
                continue

            embeddings_list.append(self.model(face.unsqueeze(0)))
            labels.append(label)

        self.caracteristicas = dict(zip(labels, embeddings_list))
        st.write(f"Se procesaron {len(embeddings_list)} imágenes.")

        if no_process_images:
            st.warning(f"No se pudieron procesar {len(no_process_images)} imágenes: {', '.join(no_process_images)}")

        return self.caracteristicas

def feature_extraction(uploaded_files):
    if not uploaded_files:
        st.warning("Por favor, carga al menos una imagen para extraer características.")
        return
    
    _models = FaceNetModels()
    if st.button("Extraer características", type="primary"):
        try:
            with st.spinner("Procesando imágenes..."):
                caracteristicas = _models.extract_embeddings(uploaded_files)
                
            if not caracteristicas:
                st.error("No se pudieron procesar las imágenes. Verifica que contengan rostros detectables.")
                return
                
            filename = f"feature_{unique_id}.pkl"

            # Crear el archivo en memoria
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_file:
                pickle.dump(caracteristicas, tmp_file)
                tmp_file.flush()
                
                # Leer el archivo para crear el download
                with open(tmp_file.name, "rb") as f:
                    pickle_data = f.read()
                
                # Usar st.download_button en lugar de HTML personalizado
                st.download_button(
                    label="📥 Descargar Características",
                    data=pickle_data,
                    file_name=f"feature_{unique_id}.pkl",
                    mime="application/octet-stream",
                    help="Descarga el archivo .pkl con las características faciales extraídas"
                )
                
                # Limpiar archivo temporal
                os.unlink(tmp_file.name)
                
            st.success("✅ Características extraídas exitosamente!")
            
        except Exception as e:
            st.error(f"❌ Ocurrió un error: {str(e)}")

def upload_and_process_image(uploaded_file, pkl_file):
    try:
        _models = FaceNetModels()

        # Guardar el archivo .pkl en una ruta temporal
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_pkl:
            temp_pkl.write(pkl_file.getvalue())  # Usar getvalue() en lugar de read()
            pkl_file_path = temp_pkl.name

        _models.load_caracteristicas(pkl_file_path)

        # Procesar la imagen subida
        img = Image.open(uploaded_file)

        if img.format == "PNG":
            img = img.convert("RGB")

        # Detectar rostro y obtener embedding
        face_tensor = _models.mtcnn(img)
        
        if face_tensor is None:
            st.error("❌ No se detectó ningún rostro en la imagen. Intenta con otra imagen.")
            os.unlink(pkl_file_path)  # Limpiar archivo temporal
            return

        image_embedding = _models.embedding(face_tensor)
        result = _models.Distancia(image_embedding)
        
        if result:
            label, distance = result
            
            # Mostrar resultados en columnas
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(img, width=200)
            
            with col2:
                st.success(f"**Persona identificada:** {label}")
                similitud = max(0, int(100 - 17.14 * distance))
                st.metric("Porcentaje de Similitud", f"{similitud}%")
                
                # Agregar indicador visual de confianza
                if similitud > 80:
                    st.success("🎯 Alta confianza")
                elif similitud > 60:
                    st.warning("⚠️ Confianza media")
                else:
                    st.error("❓ Baja confianza")
        
        # Limpiar archivo temporal
        os.unlink(pkl_file_path)

    except Exception as e:
        st.error(f"❌ Error en el procesamiento: {str(e)}")
        # Asegurar limpieza del archivo temporal
        if 'pkl_file_path' in locals():
            try:
                os.unlink(pkl_file_path)
            except:
                pass

# Configuración de la página
st.set_page_config(
    page_title="Reconocimiento Facial",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("👤 Sistema de Reconocimiento Facial")
st.markdown("---")

# Interface lateral mejorada
with st.sidebar:
    st.header("📋 Información de la App")
    
    with st.expander("ℹ️ Cómo usar la aplicación"):
        st.markdown("""
        **Esta aplicación de reconocimiento facial permite:**
        
        1. **Extraer características faciales** de un conjunto de imágenes
        2. **Guardarlas en un archivo .pkl** (diccionario de características)
        3. **Reconocer rostros** comparando con las características guardadas
        
        **Para generar características:**
        - Selecciona "Generar características"
        - Carga imágenes con nombres como: `Juan.jpg`, `Laura.jpeg`, etc.
        - Las imágenes deben contener rostros claramente visibles
        
        **Para reconocer rostros:**
        - Selecciona "Cargar diccionario y reconocer"
        - Carga el archivo .pkl generado previamente
        - Carga una imagen para identificar
        """)
    
    st.markdown("---")
    
    option = st.selectbox(
        "🎯 Seleccione una opción:",
        ("Generar características", "Cargar diccionario y reconocer"),
        help="Elige la acción que deseas realizar"
    )

# Contenido principal
if option == "Generar características":
    st.header("📸 Generar Características Faciales")
    st.markdown("Carga múltiples imágenes para extraer las características faciales de cada persona.")
    
    uploaded_files = st.file_uploader(
        "Selecciona las imágenes",
        accept_multiple_files=True,
        type=['jpg', 'jpeg', 'png'],
        help="Carga imágenes con nombres identificativos (ej: Juan.jpg, Maria.png)"
    )
    
    if uploaded_files:
        st.info(f"📊 {len(uploaded_files)} imagen(es) cargada(s)")
        
        # Mostrar preview de las imágenes
        if st.checkbox("👁️ Vista previa de imágenes"):
            cols = st.columns(min(len(uploaded_files), 4))
            for i, file in enumerate(uploaded_files[:4]):  # Mostrar máximo 4
                with cols[i % 4]:
                    img = Image.open(file)
                    st.image(img, caption=file.name, use_column_width=True)
            if len(uploaded_files) > 4:
                st.caption(f"... y {len(uploaded_files) - 4} imagen(es) más")
    
    feature_extraction(uploaded_files)

elif option == "Cargar diccionario y reconocer":
    st.header("🔍 Reconocimiento Facial")
    st.markdown("Carga un diccionario de características y una imagen para identificar a la persona.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1️⃣ Cargar Diccionario")
        pkl_file = st.file_uploader(
            "Selecciona el archivo .pkl",
            type=['pkl'],
            help="Archivo generado en el paso de extracción de características"
        )
    
    with col2:
        st.subheader("2️⃣ Cargar Imagen")
        uploaded_file = st.file_uploader(
            "Selecciona la imagen a reconocer",
            type=['jpg', 'jpeg', 'png'],
            help="Imagen que contiene el rostro a identificar"
        )
    
    if pkl_file and uploaded_file:
        st.markdown("---")
        if st.button("🚀 Iniciar Reconocimiento", type="primary"):
            with st.spinner("Analizando imagen..."):
                upload_and_process_image(uploaded_file, pkl_file)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "💡 Desarrollado con Streamlit y FaceNet PyTorch"
    "</div>", 
    unsafe_allow_html=True
)
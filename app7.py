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
            img_path = uploaded_file.name
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
            st.warning(f"No se pudieron procesar {len(no_process_images)} imágenes.")

        return self.caracteristicas


def run_feature_extraction(uploaded_files):
    _models = FaceNetModels()
    if st.button("Extraer características"):
        try:
            caracteristicas = _models.extract_embeddings(uploaded_files)
            filename = f"feature_{unique_id}.pkl"

            with open(filename, "wb") as f:
                pickle.dump(caracteristicas, f)

            with open(filename, "rb") as file:
                contents = file.read()
                base64_encoded = base64.b64encode(contents).decode("utf-8")
                download_path = f"data:application/octet-stream;base64,{base64_encoded}"
                href = f'<a href="{download_path}" download="feature_{unique_id}.pkl">Descargar Características</a>'

            st.markdown(href, unsafe_allow_html=True)

        except Exception as e:
            st.error("Ocurrió un error. Detalles: " + str(e))


def upload_and_process_image(uploaded_file, pkl_file):
    try:
        _models = FaceNetModels()

        # Guardar el archivo .pkl en una ruta temporal
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_pkl:
            temp_pkl.write(pkl_file.read())
            pkl_file_path = temp_pkl.name

        _models.load_caracteristicas(pkl_file_path)

        img = Image.open(io.BytesIO(uploaded_file.read()))

        if img.format == "PNG":
            jpg_io = io.BytesIO()
            img = img.convert("RGB")
            img.save(jpg_io, format="JPEG")
            jpg_io.seek(0)
            img = Image.open(jpg_io)

        image_embedding = _models.embedding(_models.mtcnn(img))

        result = _models.Distancia(image_embedding)
        if result:
            label, distance = result
            st.image(img, width=200)
            st.write("La imagen cargada puede ser de:", label)
            st.write("Distancia Euclidiana: ", round(distance, 4))

            show_recognized_face(label, dir_img)

            # img_files = os.listdir(data_dir)
            # st.write(str(img_files))

            # for img_file in img_files:
            #     img_path = os.path.join(data_dir, img_file)
            #     img_label = os.path.splitext(img_file)[0]
            #     if (
            #         img_label == label
            #     ):  # Comparación de etiquetas sin distinción de mayúsculas y minúsculas
            #         image = Image.open(img_path)
            #         st.image(image, caption="Imagen del rostro reconocido", width=200)
            #     # return

            # st.write("No se encontró la imagen correspondiente al rostro reconocido.")

        else:
            st.write(
                "Algo falló con la imagen proporcionada. Verifica si la ruta del diccionario de Caracteristicas es correcta "
                + "O si el mismo ha sido generado previamente."
            )

    except Exception as e:
        print("Error en upload_and_process_image:", str(e))
        return None


def show_recognized_face(label, dir):
    img_files = os.listdir(dir)

    for img_file in img_files:
        img_path = os.path.join(dir, img_file)
        img_label = os.path.splitext(img_file)[0]
        if (
            img_label == label
        ):  # Comparación de etiquetas sin distinción de mayúsculas y minúsculas
            image = Image.open(img_path)
            st.image(image, caption="Imagen del rostro reconocido", width=200)
            return

    st.write("No se encontró la imagen correspondiente al rostro reconocido.")


# Interface lateral

expander = st.expander("Información de la App")
with expander:
    st.write("Esta es una aplicación de reconocimiento facial.")
    st.write(
        "Permite extraer características faciales, guardarlas en un archivo (diccionario con extencion .pkl) y luego reconocer el rostro en una imagen."
    )
    st.write(
        "Para generar características faciales, selecciona la opción 'Generar características' en el menú lateral y especifica el directorio de trabajo donde se alojan las imagenes."
    )
    st.write(
        "Una vez generadas las características, puedes cargar el diccionario .pkl y una imagen para realizar el reconocimiento facial."
    )


dir_img = st.sidebar.text_input("Directorio de imagenes")

# st.sidebar.title("Opciones")
option = st.sidebar.selectbox(
    "Seleccione una opción:",
    ("Generar características", "Cargar diccionario y reconocer"),
)

if option == "Generar características":
    uploaded_files = st.file_uploader("Cargar imágenes", accept_multiple_files=True)
    run_feature_extraction(uploaded_files)
elif option == "Cargar diccionario y reconocer":
    # data_dir = st.sidebar.text_input("Directorio de trabajo")
    pkl_file = st.file_uploader("Cargar archivo .pkl")
    uploaded_file = st.file_uploader("Cargar imagen a reconocer")
    if pkl_file and uploaded_file:
        upload_and_process_image(uploaded_file, pkl_file)

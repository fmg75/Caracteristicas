import os
import io
import streamlit as st
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import pickle
import uuid

# Generar un ID único utilizando uuid
unique_id = str(uuid.uuid4())[:8]
data_dir = st.text_input("Ingrese ruta de carpeta de trabajo:")


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

    def extract_embeddings(self, data_dir):
        embeddings_list = []
        labels = []
        no_process_dir = os.path.join(data_dir, "no_process")
        os.makedirs(no_process_dir, exist_ok=True)

        img_files = os.listdir(data_dir)
        for img_file in img_files:
            img_path = os.path.join(data_dir, img_file)
            # check the file is an image or not
            if os.path.splitext(img_path)[1].lower() in (".jpg", ".jpeg", ".png"):
                img = Image.open(img_path)
                img = img.convert("RGB")
                # Obtener el nombre del archivo de la imagen actual
                label = os.path.splitext(os.path.basename(img_path))[0]
                face = self.mtcnn(img)
                if face is None:
                    img.save(
                        os.path.join(
                            no_process_dir, os.path.splitext(img_file)[0] + ".png"
                        )
                    )
                    continue
                embeddings_list.append(self.model(face.unsqueeze(0)))
                labels.append(label)

        self.caracteristicas = dict(zip(labels, embeddings_list))

        filename = os.path.join(data_dir, f"feature_{unique_id}.pkl")

        # Serializar la variable caracteristicas y guardarla en el archivo con el nombre generado
        with open(filename, "wb") as f:
            pickle.dump(self.caracteristicas, f)

        # Mostrar la cantidad de imágenes procesadas y no procesadas
        st.write(f"Se procesaron {len(embeddings_list)} imágenes.")
        no_process_files = os.listdir(no_process_dir)
        if no_process_files:
            st.warning(
                f"No se pudieron procesar {len(no_process_files)} imágenes "
                f"Revise la carpeta '{no_process_dir}' para ver los archivos no procesados."
            )
        return self.caracteristicas


def process_image(path, data_dir):
    try:
        _models = FaceNetModels()
        # Obtener la lista de archivos .pkl en el directorio de características
        file_list = [file for file in os.listdir(data_dir) if file.endswith(".pkl")]

        if len(file_list) == 0:
            st.write("No se encontraron archivos .pkl en el directorio de trabajo.")
            st.write("Genere el diccionario de caracteristicas!")
            return None

        # Cargar el primer archivo .pkl encontrado
        filename = os.path.join(data_dir, file_list[0])

        _models.load_caracteristicas(filename)

        img = Image.open(path)

        # Verificar si la imagen está en formato PNG y convertir a JPG si es necesario
        if img.format == "PNG":
            jpg_io = (
                io.BytesIO()
            )  # Crear un objeto BytesIO para guardar la imagen en memoria
            img = img.convert(
                "RGB"
            )  # Convertir a modo RGB (requerido para guardar como JPG)
            img.save(
                jpg_io, format="JPEG"
            )  # Guardar la imagen en el objeto BytesIO en formato JPG
            jpg_io.seek(0)  # Colocar el puntero de lectura al inicio del objeto BytesIO
            img = Image.open(
                jpg_io
            )  # Abrir la imagen en formato JPG desde el objeto BytesIO

        image_embedding = _models.embedding(_models.mtcnn(img))

        return _models.Distancia(image_embedding)

    except Exception as e:
        print("Error en process_image:", str(e))

        return None


def show_recognized_face(label, data_dir):
    img_files = os.listdir(data_dir)
    for img_file in img_files:
        img_path = os.path.join(data_dir, img_file)
        img_label = os.path.splitext(img_file)[
            0
        ]  # Etiqueta de la imagen (sin extensión)
        if img_label == label:
            image = Image.open(img_path)
            st.image(image, caption="Imagen del rostro reconocido", width=200)
            return

    st.write("No se encontró la imagen correspondiente al rostro reconocido.")


def upload_image():
    uploaded_file = st.file_uploader(
        "Subir la imagen de un Rostro",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen subida ", width=200)
        # _models = FaceNetModels()
        result = process_image(uploaded_file, data_dir)
        if result:
            label, distance = result
            st.write("La imagen cargada puede ser de:", label)
            st.write("Distancia Euclidiana: ", round(distance, 4))
            show_recognized_face(label, data_dir)
        else:
            st.write(
                "Algo falló con la imagen proporcionada, intenta con otra!!"
                + "Verifica si la ruta del diccionario de Caracteristicas es correcta "
                + "O si el mismo a sido generado previamente"
            )


# def run_feature_extraction():
#     if data_dir and os.path.isdir(data_dir):
#         _models = FaceNetModels()
#         if st.button("Extraer características"):
#             caracteristicas = _models.extract_embeddings(data_dir)
#             st.write("Diccionario de características:")
#             st.write(caracteristicas)

#     else:
#         st.warning("Ingrese una ruta de carpeta válida para continuar.")


def run_feature_extraction():
    _models = FaceNetModels()
    if st.button("Extraer características"):
        try:
            caracteristicas = _models.extract_embeddings(data_dir)
            st.write("Diccionario de características:")
            st.write(caracteristicas)
        except Exception as e:
            st.error("Ingrese una ruta Valida: ")


# Crear una barra lateral para seleccionar la página
page = st.sidebar.selectbox(
    "Seleccione una página", ["Extracción de Características", "Reconocedor de Rostros"]
)

# Verificar la página seleccionada y mostrar el contenido correspondiente
if page == "Reconocedor de Rostros":
    # Código del segundo archivo (actual)
    st.markdown(
        "<h1 style='color: green;'>RECONOCEDOR DE ROSTROS</h1>", unsafe_allow_html=True
    )
    upload_image()
elif page == "Extracción de Características":
    # Código del primer archivo
    st.markdown(
        "<h1 style='color: green;'>EXTRACCION CARACTERISTICAS</h1>",
        unsafe_allow_html=True,
    )
    run_feature_extraction()

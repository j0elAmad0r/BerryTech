import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import av
import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from inference_sdk import (
    InferenceConfiguration,
    InferenceHTTPClient,
)
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer


WORKSPACE = "amadors-workspace"
WORKFLOW = "detector-de-fresas"

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "resultados"
HISTORY_FILE = BASE_DIR / "historial.json"

RESULTS_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="BerryTech",
    page_icon="🍓",
    layout="wide",
)


def load_history():
    if not HISTORY_FILE.exists():
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        return []


def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4, ensure_ascii=False)


def get_client():
    api_key = os.environ.get("ROBOFLOW_API_KEY")

    if not api_key:
        raise RuntimeError(
            "No se encontró la variable ROBOFLOW_API_KEY."
        )

    return InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key,
    ).configure(
        InferenceConfiguration(
            api_key_transport="header",
        )
    )


def extract_predictions(result):
    output = result[0] if isinstance(result, list) else result
    predictions = output.get("predictions", [])

    if isinstance(predictions, dict):
        predictions = predictions.get("predictions", [])

    return predictions


def draw_predictions(image, predictions):
    result_image = image.copy()

    for prediction in predictions:
        x = prediction.get("x", 0)
        y = prediction.get("y", 0)
        width = prediction.get("width", 0)
        height = prediction.get("height", 0)

        x1 = int(x - width / 2)
        y1 = int(y - height / 2)
        x2 = int(x + width / 2)
        y2 = int(y + height / 2)

        label = prediction.get("class", "Fresa")
        confidence = prediction.get("confidence", 0)

        color = (0, 255, 0)

        cv2.rectangle(
            result_image,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        cv2.putText(
            result_image,
            f"{label} {confidence:.1%}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    cv2.putText(
        result_image,
        f"Fresas detectadas: {len(predictions)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
    )

    return result_image


def get_statistics(history):
    total_strawberries = 0
    healthy = 0
    symptoms = 0
    pending = 0
    confidences = []

    for analysis in history:
        total_strawberries += analysis.get(
            "fresas_detectadas",
            0,
        )

        for prediction in analysis.get("predicciones", []):
            label = prediction.get("class", "").lower()
            confidence = prediction.get("confidence", 0)

            confidences.append(confidence)

            if "salud" in label or "healthy" in label:
                healthy += 1
            elif "sintoma" in label or "symptom" in label:
                symptoms += 1

            if confidence < 0.6:
                pending += 1

    return {
        "total": total_strawberries,
        "healthy": healthy,
        "symptoms": symptoms,
        "pending": pending,
        "confidences": confidences,
    }


class StrawberryVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.last_inference = 0
        self.predictions = []
        self.client = get_client()

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        current_time = time.time()

        if current_time - self.last_inference >= 10:
            try:
                temporary_path = BASE_DIR / "captura_video.jpg"
                cv2.imwrite(str(temporary_path), image)

                response = self.client.run_workflow(
                    workspace_name=WORKSPACE,
                    workflow_id=WORKFLOW,
                    images={"image": str(temporary_path)},
                    use_cache=True,
                )

                with self.lock:
                    self.predictions = extract_predictions(response)

                self.last_inference = current_time

            except Exception:
                with self.lock:
                    self.predictions = []

                self.last_inference = current_time

        with self.lock:
            predictions = list(self.predictions)

        annotated_image = draw_predictions(
            image,
            predictions,
        )

        return av.VideoFrame.from_ndarray(
            annotated_image,
            format="bgr24",
        )


def show_dashboard():
    history = load_history()
    statistics = get_statistics(history)

    st.title("🍓 BerryTech")
    st.caption("AGROVISION AI · PANEL DE CONTROL")

    st.header("Panel principal")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Fresas detectadas",
            statistics["total"],
        )

    with col2:
        st.metric(
            "Saludables",
            statistics["healthy"],
        )

    with col3:
        st.metric(
            "Con síntomas",
            statistics["symptoms"],
        )

    with col4:
        st.metric(
            "Pendientes de revisión",
            statistics["pending"],
        )

    st.divider()

    if history:
        last_analysis = history[-1]

        st.subheader("Último análisis")
        st.write(
            f"Archivo: **{last_analysis.get('archivo', 'Desconocido')}**"
        )
        st.write(
            f"Fecha: **{last_analysis.get('fecha', 'Desconocida')}**"
        )
    else:
        st.info(
            "Todavía no hay análisis. "
            "Ve a 'Analizar imagen' para comenzar."
        )
    st.divider()
    st.subheader("Cámara en tiempo real")

    activar_camara = st.checkbox(
        "Activar cámara",
        key="activar_camara_principal",
    )
    if activar_camara:
        show_realtime_camera()

def show_image_analysis():
    st.title("Analizar imagen")

    uploaded_file = st.file_uploader(
        "Selecciona una imagen",
        type=["jpg", "jpeg", "png"],
    )

    if not uploaded_file:
        st.info("Selecciona una imagen para comenzar.")
        return

    image_bytes = uploaded_file.read()
    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )
    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        st.error("No se pudo abrir la imagen.")
        return

    if st.button(
        "Analizar imagen",
        type="primary",
    ):
        temporary_path = BASE_DIR / "captura_actual.jpg"
        cv2.imwrite(str(temporary_path), image)

        with st.spinner("Analizando con Roboflow..."):
            try:
                client = get_client()

                response = client.run_workflow(
                    workspace_name=WORKSPACE,
                    workflow_id=WORKFLOW,
                    images={"image": str(temporary_path)},
                    use_cache=True,
                )

                predictions = extract_predictions(response)
                result_image = draw_predictions(
                    image,
                    predictions,
                )

                timestamp = datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )

                result_path = RESULTS_DIR / (
                    f"resultado_{timestamp}.jpg"
                )

                cv2.imwrite(
                    str(result_path),
                    result_image,
                )

                history = load_history()

                history.append(
                    {
                        "fecha": datetime.now().isoformat(),
                        "archivo": uploaded_file.name,
                        "fresas_detectadas": len(predictions),
                        "predicciones": predictions,
                        "resultado": str(result_path),
                    }
                )

                save_history(history)

                st.session_state["result_image"] = result_image
                st.session_state["predictions"] = predictions

                st.success("Análisis completado.")

            except Exception as error:
                st.error(f"Error al analizar la imagen: {error}")

    if "result_image" not in st.session_state:
        return

    result_image = st.session_state["result_image"]
    predictions = st.session_state["predictions"]

    st.subheader("Resultado")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            cv2.cvtColor(
                result_image,
                cv2.COLOR_BGR2RGB,
            ),
            use_container_width=True,
        )

    with col2:
        st.metric(
            "Fresas detectadas",
            len(predictions),
        )

        if predictions:
            confidences = [
                prediction.get("confidence", 0)
                for prediction in predictions
            ]

            st.subheader("Confianza")

            figure, axis = plt.subplots()

            axis.bar(
                range(1, len(confidences) + 1),
                confidences,
                color="#3A9D5D",
            )

            axis.set_ylim(0, 1)
            axis.set_xlabel("Detección")
            axis.set_ylabel("Confianza")
            axis.grid(axis="y", alpha=0.3)

            st.pyplot(
                figure,
                clear_figure=True,
            )
        else:
            st.info("No se detectaron fresas.")


def show_realtime_camera():
    st.title("Cámara en tiempo real")

    st.info(
        "La cámara analiza un frame aproximadamente cada 10 segundos."
    )

    try:
        webrtc_streamer(
            key="berrytech-camera",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=StrawberryVideoProcessor,
            media_stream_constraints={
                "video": True,
                "audio": False,
            },
            async_processing=True,
        )
    except Exception as error:
        st.error(
            f"No se pudo iniciar la cámara: {error}"
        )


def show_history():
    st.title("Historial de análisis")

    history = load_history()

    if not history:
        st.info("No hay análisis guardados.")
        return

    for analysis in reversed(history):
        with st.expander(
            f"{analysis.get('archivo', 'Sin nombre')} · "
            f"{analysis.get('fecha', '')}"
        ):
            st.write(
                f"Fresas detectadas: "
                f"**{analysis.get('fresas_detectadas', 0)}**"
            )
            st.write(
                f"Resultado guardado en: "
                f"`{analysis.get('resultado', '')}`"
            )


def show_statistics():
    st.title("Estadísticas")

    history = load_history()
    statistics = get_statistics(history)

    if not history:
        st.info("Todavía no hay análisis guardados.")
        return

    st.metric(
        "Total de análisis",
        len(history),
    )

    confidences = statistics["confidences"]

    if confidences:
        st.subheader("Distribución de confianza")

        figure, axis = plt.subplots()

        axis.hist(
            confidences,
            bins=10,
            range=(0, 1),
            color="#3A9D5D",
            edgecolor="white",
        )

        axis.set_xlabel("Confianza")
        axis.set_ylabel("Detecciones")
        axis.grid(axis="y", alpha=0.3)

        st.pyplot(
            figure,
            clear_figure=True,
        )

    st.divider()
    st.subheader("Historial de análisis")

    for analysis in reversed(history):
        with st.expander(
            f"{analysis.get('archivo', 'Sin nombre')} · "
            f"{analysis.get('fecha', '')}"
        ):
            st.write(
                f"Fresas detectadas: "
                f"**{analysis.get('fresas_detectadas', 0)}**"
            )

            st.write(
                f"Resultado: "
                f"`{analysis.get('resultado', '')}`"
            )


st.sidebar.title("🍓 BerryTech")
st.sidebar.caption("Sistema activo")

page = st.sidebar.radio(
    "Navegación",
    [
        "Panel principal",
        "Analizar imagen",
        "Estadísticas",
    ],
)

if page == "Panel principal":
    show_dashboard()
elif page == "Analizar imagen":
    show_image_analysis()
elif page == "Estadísticas":
    show_statistics()
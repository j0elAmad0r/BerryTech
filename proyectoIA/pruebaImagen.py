
import cv2
import os
import json
import tkinter as tk

from tkinter import filedialog
from inference_sdk import (
    InferenceHTTPClient,
    InferenceConfiguration
)

# Conectar con Roboflow
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.environ["ROBOFLOW_API_KEY"]
).configure(
    InferenceConfiguration(
        api_key_transport="header"
    )
)

WORKSPACE = "amadors-workspace"
WORKFLOW = "detector-de-fresas"

# Ventana para seleccionar una imagen
root = tk.Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Seleccionar imagen para analizar",
    filetypes=[
        ("Imágenes", "*.jpg *.jpeg *.png")
    ]
)

if not image_path:
    print("No se seleccionó ninguna imagen.")
    exit()

# Cargar imagen
image = cv2.imread(image_path)

if image is None:
    print("No se pudo abrir la imagen.")
    exit()

print("Imagen seleccionada:", image_path)

# Enviar imagen a Roboflow
try:
    result = client.run_workflow(
        workspace_name=WORKSPACE,
        workflow_id=WORKFLOW,
        images={"image": image_path},
        use_cache=True
    )

    output = result[0] if isinstance(result, list) else result

    predictions = output.get(
        "predictions", {}
    ).get("predictions", [])

    print("\nRESULTADOS DE ROBOFLOW:")
    print(json.dumps(result, indent=4))

    print("\nFresas detectadas:", len(predictions))

    # Dibujar cuadros de detección
    for prediction in predictions:

        x = prediction["x"]
        y = prediction["y"]
        w = prediction["width"]
        h = prediction["height"]

        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        label = prediction.get("class", "Fresa")
        confidence = prediction.get("confidence", 0)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        text = f"{label} {confidence:.1%}"

        cv2.putText(
            image,
            text,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Guardar y mostrar resultado
    cv2.imwrite("resultado_deteccion.jpg", image)

    cv2.imshow("BerryTech - Analisis de imagen", image)

    print("\nPresiona cualquier tecla para cerrar.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as error:
    print("Error al analizar la imagen:", error)
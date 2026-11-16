# BerryTech
import cv2
import os
import time
import json

from inference_sdk import (
    InferenceHTTPClient,
    InferenceConfiguration
)

# Configuracion de Roboflow
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

# Inicializar camara
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error al abrir la camara")
    exit()

print("BerryTech AgroVision AI iniciado")
print("Presiona ESC para salir")

last_inference = 0
interval = 10
predictions = []

while True:

    ret, frame = camera.read()

    if not ret:
        print("Error al capturar imagen")
        break

    current_time = time.time()

    # Enviar imagen a Roboflow cada 10 segundos
    if current_time - last_inference >= interval:

        try:
            # Guardar captura temporal
            image_path = "captura_actual.jpg"
            # Ejecutar Workflow
            cv2.imwrite(image_path, frame)

            print("Imagen guardada:", image_path)
            print("Resolucion de la camara:", frame.shape)

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

            last_inference = current_time

            print("\nRESULTADOS DE ROBOFLOW:")
            print(json.dumps(result, indent=4))


        except Exception as error:
            print("Error en Roboflow:", error)
            predictions = []
            last_inference = current_time


    for prediction in predictions:
        x = prediction["x"]
        y = prediction["y"]
        w = prediction["width"]
        h = prediction["height"]

        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        confidence = prediction.get("confidence", 0)
        label = prediction.get("class", "Fresa")

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} {confidence:.1%}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        f"Fresas detectadas: {len(predictions)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )    
    # Mostrar camara
    cv2.imshow(
        "BerryTech AgroVision AI",
        frame
    )

    if cv2.waitKey(1) == 27:
        break


camera.release()
cv2.destroyAllWindows()
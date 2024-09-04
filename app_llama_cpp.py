import base64
import json
import os, time
from llama_cpp import Llama
from llama_cpp.llama_chat_format import MiniCPMv26ChatHandler


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


# Path to your image
image_path = "images/PXL_20240814_200220044.jpg"

# Getting the base64 string
image = encode_image(image_path)

chat_handler = MiniCPMv26ChatHandler(clip_model_path="models/mmproj-model-f16.gguf")
model = Llama(model_path="models/ggml-model-Q4_K.gguf", chat_handler=chat_handler, n_ctx=2048, n_gpu_layers=-1)

prompt = """
Lese den Zählerstand des abgebildeten Drehstromzählers ab.
Der Zählerstand besteht aus sechs Ziffern und einer Nachkommastelle.
Die Zahl ist im Bereich von 46600 und 46700.
Die Einheit ist in kWh.

Gib das Ergebnis als JSON-Ausdruck in folgenden Format wider:
{"wert": zahl, "einheit": "physikalische Einheit des Wertes"}.
Bevor du das Ergebnis ausgibst, stelle sicher, dass der Wert korrekt ist und vollständig erfasst wird.
"""

prompt_blutdruck = """
Du siehst hier ein Blutdruck-Messgerät mit 3 Werten: Puls, systolischer Blutdruck und diastolischer Blutdruck.
Lese nur den systolischen Blutdruck von dem Blutdruck-Messgerät ab. Das ist der Wert, wo "sys" steht.
Die Einheit ist in mmHg.

Gib das Ergebnis als JSON-Ausdruck in folgender Form wider:
{"wert": "zahl als string", "einheit": "physikalische Einheit des Wertes"}.

Bevor du das Ergebnis ausgibst, stelle sicher, dass der Wert korrekt und vollständig erfasst wird.
"""

t0 = time.time()
chat_completion = model.create_chat_completion(
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image}"
                    }
                }
                ]
        }
    ],
    response_format={"type": "json_object"},
    temperature=0.0
)
print(time.time() - t0)

result = json.loads(chat_completion["choices"][0]["message"]["content"])

print(result)

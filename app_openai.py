import base64
import json
import os
from openai import OpenAI


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


# Path to your image
image_path = "images/PXL_20240826_102503113.jpg"

# Getting the base64 string
image = encode_image(image_path)

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

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

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_blutdruck
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
    model="gpt-4o",
    response_format={"type": "json_object"},
    temperature=0.0
)

result = json.loads(chat_completion.choices[0].message.content)

print(type(result))
print(result)

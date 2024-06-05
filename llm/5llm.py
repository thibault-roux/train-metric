import os
from openai import OpenAI
import pickle

ihavemoney = True

if ihavemoney:
    client = OpenAI(
        api_key = os.environ.get("OPENAI_API_KEY"),
    )


    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Tu écris seulement 'A' ou 'B' pour choisir la meilleure transcription."},
        {"role": "user", "content": "Référence : Mon radiateur est en panne\nHypothèse A : Mon mon radiateur est en panne\nHypothèse B : Mon radieux pend"},
        {"role": "assistant", "content": "A"},
    ]
    )

    pickle.dump(response, open("response.pkl", "wb"))
else:
    response = pickle.load(open("response.pkl", "rb"))


print(response)
print()
print(response.choices[0].message.content)

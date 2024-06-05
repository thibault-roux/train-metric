import os
from openai import OpenAI
import pickle


# source ~/.zshrc

def chat(ref, hypA, hypB):
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Tu écris seulement 'A' ou 'B' pour choisir la meilleure transcription."},
        {"role": "user", "content": "Référence : Mon radiateur est en panne\nHypothèse A : Mon mon radiateur est en panne\nHypothèse B : Mon radieux pend"},
        {"role": "assistant", "content": "A"},
        {"role": "user", "content": "Référence : " + ref + "\nHypothèse A : " + hypA + "\nHypothèse B : " + hypB},
    ]
    )
    pickle.dump(response, open("response2.pkl", "wb"))


ihavemoney = True
if ihavemoney:
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    ref = "c' est à lui même"
    hypA = "êtes à lui même"
    hypB = "c' est euh à lui-même"
    chat(ref, hypA, hypB)

response = pickle.load(open("response2.pkl", "rb"))


print(response)
print()
print(response.choices[0].message.content)

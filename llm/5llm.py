import os
from openai import OpenAI
import pickle

ihavemoney = False

if ihavemoney:
    client = OpenAI(
        api_key = "",
    )


    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]
    )

    pickle.dump(response, open("response.pkl", "wb"))
else:
    response = pickle.load(open("response.pkl", "rb"))


print(response)
print()
print(response.choices[0].message.content)

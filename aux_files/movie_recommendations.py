from dotenv import load_dotenv, find_dotenv
import json
import os
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv

load_dotenv("openAI.env")

client = OpenAI(api_key=os.environ["openai_apikey"])



with open('movie_descriptions_embeddings.json', 'r') as file:
    file_content = file.read()
    movies = json.loads(file_content)

# Esta función devuelve un embedding de un texto
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

# Similitud de coseno con NumPy
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Ejemplo de recomendación
req = "película de un pianista"
emb = get_embedding(req)

sim = []
for i in range(len(movies)):
    sim.append(cosine_similarity(emb, movies[i]['embedding']))

sim = np.array(sim)
idx = np.argmax(sim)
print(movies[idx]['title'])

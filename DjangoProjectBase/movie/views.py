from django.shortcuts import render
from django.http import HttpResponse

from .models import Movie
import sys, os
import matplotlib.pyplot as plt
import matplotlib
import io
import urllib, base64
from openai import OpenAI
import numpy as np
import os
from dotenv import load_dotenv
from django.shortcuts import render
from django.conf import settings



def home(request):
    #return HttpResponse('<h1>Welcome to Home Page</h1>')
    #return render(request, 'home.html')
    #return render(request, 'home.html', {'name':'Paola Vallejo'})
    searchTerm = request.GET.get('searchMovie') # GET se usa para solicitar recursos de un servidor
    if searchTerm:
        movies = Movie.objects.filter(title__icontains=searchTerm)
    else:
        movies = Movie.objects.all()
    return render(request, 'home.html', {'searchTerm':searchTerm, 'movies':movies})


def recommendationsSystem(request): 
    prompt = request.GET.get('prompt')
    recommended_movies = []

    if prompt:
        # ✅ Obtener el embedding del prompt
        prompt_embedding = get_embedding(prompt)

        # ✅ Cargar películas con embeddings
        all_movies = Movie.objects.exclude(embedding__isnull=True)

        similarities = []
        for movie in all_movies:
            try:
                movie_embedding = np.frombuffer(movie.embedding, dtype=np.float32)
                similarity = cosine_similarity(prompt_embedding, movie_embedding)
                similarities.append((similarity, movie))
            except Exception as e:
                print(f"Error with movie {movie.title}: {e}")

        # ✅ Ordenar películas por similitud
        similarities.sort(reverse=True, key=lambda x: x[0])

        # ✅ Tomar las top 5
        recommended_movies = [movie for _, movie in similarities[:5]]
        print(recommended_movies)

    return render(request, 'recommendations.html', {
        'prompt': prompt,
        'recommended_movies': recommended_movies
    })


def about(request):
    #return HttpResponse('<h1>Welcome to About Page</h1>')
    return render(request, 'about.html')

def signup(request):
    email = request.GET.get('email') 
    return render(request, 'signup.html', {'email':email})


def statistics_view0(request):
    matplotlib.use('Agg')
    # Obtener todas las películas
    all_movies = Movie.objects.all()

    # Crear un diccionario para almacenar la cantidad de películas por año
    movie_counts_by_year = {}

    # Filtrar las películas por año y contar la cantidad de películas por año
    for movie in all_movies:
        year = movie.year if movie.year else "None"
        if year in movie_counts_by_year:
            movie_counts_by_year[year] += 1
        else:
            movie_counts_by_year[year] = 1

    # Ancho de las barras
    bar_width = 0.5
    # Posiciones de las barras
    bar_positions = range(len(movie_counts_by_year))

    # Crear la gráfica de barras
    plt.bar(bar_positions, movie_counts_by_year.values(), width=bar_width, align='center')

    # Personalizar la gráfica
    plt.title('Movies per year')
    plt.xlabel('Year')
    plt.ylabel('Number of movies')
    plt.xticks(bar_positions, movie_counts_by_year.keys(), rotation=90)

    # Ajustar el espaciado entre las barras
    plt.subplots_adjust(bottom=0.3)

    # Guardar la gráfica en un objeto BytesIO
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    # Convertir la gráfica a base64
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png)
    graphic = graphic.decode('utf-8')

    # Renderizar la plantilla statistics.html con la gráfica
    return render(request, 'statistics.html', {'graphic': graphic})

def statistics_view(request):
    matplotlib.use('Agg')
    # Gráfica de películas por año
    all_movies = Movie.objects.all()
    movie_counts_by_year = {}
    for movie in all_movies:
        print(movie.genre)
        year = movie.year if movie.year else "None"
        if year in movie_counts_by_year:
            movie_counts_by_year[year] += 1
        else:
            movie_counts_by_year[year] = 1

    year_graphic = generate_bar_chart(movie_counts_by_year, 'Year', 'Number of movies')

    # Gráfica de películas por género
    movie_counts_by_genre = {}
    for movie in all_movies:
        # Obtener el primer género
        genres = movie.genre.split(',')[0].strip() if movie.genre else "None"
        if genres in movie_counts_by_genre:
            movie_counts_by_genre[genres] += 1
        else:
            movie_counts_by_genre[genres] = 1

    genre_graphic = generate_bar_chart(movie_counts_by_genre, 'Genre', 'Number of movies')

    return render(request, 'statistics.html', {'year_graphic': year_graphic, 'genre_graphic': genre_graphic})


def generate_bar_chart(data, xlabel, ylabel):
    keys = [str(key) for key in data.keys()]
    plt.bar(keys, data.values())
    plt.title('Movies Distribution')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png).decode('utf-8')
    return graphic



# ✅ Cargar la API key de OpenAI

#load_dotenv('../openAI.env')
#client = OpenAI(api_key=os.environ.get('openai_apikey'))

# ✅ Función para generar embeddings
def get_embedding(text, model="text-embedding-3-small"):
    import os
    from dotenv import load_dotenv
    from openai import OpenAI
    import numpy as np

    # Cargar archivo .env (si no se ha hecho)
    load_dotenv('../openAI.env')

    # Obtener API key del entorno
    api_key = os.getenv('openai_apikey')
    if not api_key:
        raise ValueError("❌ La variable de entorno 'openai_apikey' no está definida.")

    # Crear cliente OpenAI
    client = OpenAI(api_key=api_key)

    # Limpiar el texto y obtener embeddings
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding, dtype=np.float32)

# ✅ Función para similitud de coseno
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

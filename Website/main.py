import re
from flask import Flask
from flask import render_template
from flask import request
import pandas as pd 
import math
import pickle as pk
import numpy as np


app=Flask(__name__)


movies_data=pd.read_csv('data/final_movie_data.csv',encoding='latin-1')
movie_contents=pd.read_csv('data/final_contents.csv',encoding='latin-1')


with open('Model/knnpickle_file_content.pkl', 'rb') as file:
    knn_model=pk.load(file)





def recommend_on_movie_content_based(movie,n_reccomend = 5):
    input_movie_index=movies_data[movies_data['title'].str.strip()==movie].index[0]
    input_movie_generes=movie_contents.iloc[input_movie_index]
    
    _,neighbors=knn_model.kneighbors([input_movie_generes],n_neighbors=n_reccomend+1)
    
    recommeds_movies=[]
    recommends_rate=[]
    for i in neighbors[0]:
        if i !=input_movie_index:
            recommeds_movies.append(movies_data.iloc[i]['title'])
            recommends_rate.append((movies_data.iloc[i]['rate']))
    
    recommends_rate,recommeds_movies= zip(*sorted(zip(recommends_rate,recommeds_movies)))
    
    
    
    return [recommeds_movies[-1-i]+'- '+str(recommends_rate[-1-i]) for i in range(0,n_reccomend)]


links=[]


data=['Cocaine Bear (2023)','Operation Fortune (2023)','The Last Kingdom(2023)','American Psycho (2000)']
links=['static/assets/movie9.webp','static/assets/movie10.webp','static/assets/movie11.webp','static/assets/movie12.webp']
r_links=['static/assets/movie13.webp','static/assets/movie14.webp','static/assets/movie15.webp','static/assets/movie16.webp']

@app.route('/', methods =["GET", "POST"])
def gfg():
    if request.method == "POST":
       movie_name = request.form.get("movie_name")

       if movie_name !="":
            res=recommend_on_movie_content_based(movie_name.strip())
            return render_template("base.html",data=res,links=r_links)
       
    return render_template("base.html",data=data,links=links)
 


@app.route('/about')
def about():

     return render_template('about.html')


@app.route('/')
def index():

    return render_template('base.html')        
    

if __name__=='__main__':
    app.run(debug=True)
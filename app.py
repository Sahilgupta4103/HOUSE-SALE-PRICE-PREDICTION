import os
import pandas as pd 
import numpy as np 
import flask
import pickle
import joblib
from flask import Flask, render_template, request
import requests

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def home():
 return render_template('index.html')

@app.route("/predict",methods = ["GET","POST"])
def result():
 if request.method == "POST":
   date = int(request.form.get('date'))
   bedrooms = float(request.form.get('bedrooms'))
   bathrooms = float(request.form.get('bathrooms'))
   sqft_living = int(request.form.get('sqft_living'))
   sqft_lot = int(request.form.get('sqft_lot'))
   floors = float(request.form.get('floors'))
   waterfront = int(request.form.get('waterfront'))
   view = float(request.form.get('view'))
   condition = int(request.form.get('condition'))
   grade = int(request.form.get('grade'))
   sqft_above = int(request.form.get('sqft_above'))
   sqft_basement = int(request.form.get('sqft_basement'))
   yr_built = int(request.form.get('yr_built'))
   yr_renovated = int(request.form.get('yr_renovated'))
   zipcode = int(request.form.get('zipcode'))
   lat = float(request.form.get('lat'))
   long = float(request.form.get('long'))
   sqft_living15 = int(request.form.get('sqft_living15'))
   sqft_lot15 = int(request.form.get('sqft_lot15'))
   prediction = [date, bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, grade, sqft_above, sqft_basement, yr_built, yr_renovated, zipcode, lat, long, sqft_living15, sqft_lot15]
   prediction = np.array(prediction)
   prediction = prediction.reshape(1, -1)    
   file = open("model.pkl","rb")
   trained_model = joblib.load(file)
   result = trained_model.predict(prediction)
   return render_template("index.html", result = str(result[0]))

if __name__ == "__main__":
 app.run(debug=True)
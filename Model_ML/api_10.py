#import library
import flask
from flask import Flask, request, jsonify
import joblib, warnings, os, requests
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics.pairwise import haversine_distances
warnings.filterwarnings('ignore')

print(flask.__version__)
print(joblib.__version__)
print(requests.__version__)
print(pd.__version__)
print(np.__version__)
print(sklearn.__version__)

app = Flask(__name__)


@app.route('/v1/predict/<Longitude>/<Latitude>', methods=["GET"])
def recommended_hospitals(Longitude, Latitude):
  """
  Predict the class and recommended_hospitals
  """
  #load dataset
  model = pickle.load(open('....', 'rb'))
  mean = pickle.load(open('....', 'rb'))
  std = pickle.load(open('....', 'rb'))
  data = pickle.load(open('....', 'rb'))

  #mengubah inputan mejadi dataframe
  df_input = pd.DataFrame({"Longitude": [Longitude], "Latitude": [Latitude]})

  #fiture scaling
  scaled = pd.DataFrame(tf.divide(tf.subtract(df_input, mean), std))

  #Prediksi cluster
  Cluster = model.predict(tf.reshape(scaled, (1, -1)))[0]
  data_cluster = data[data['Cluster']==Cluster].copy()
  print(Cluster)

  data_cluster.reset_index(inplace = True, drop = True)
  inverse = lambda x: np.sum([tf.multiply(scaled, std), mean])
  tmp = data_cluster.apply(inverse, axis = 1)
  tmp_long = [tmp[x][0][0] for x in range(len(tmp))];
  tmp_lat = [tmp[x][0][1] for x in range(len(tmp))];
  data_cluster.Longitude = np.array(tmp_long); data_cluster.Latitude = np.array(tmp_lat);

  #Mengurutkan hasil rekomendasi
  col_name = ['Longitude', 'Latitude']
  data_cluster.sort_values(col_name, ascending = [True]*len(col_name), inplace = True)

  #Pilih jumlah N rekomendasi
  n = 50
  data_cluster = data_cluster.iloc[:n]
  return data_clusterer
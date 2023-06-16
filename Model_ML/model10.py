# **Import Library**
"""

import csv
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.cluster import KMeans
from sklearn import metrics

#Load Dataset
filename_df = pd.read_csv("Data Rumah Sakit di Surakarta - Sheet1 (1).csv")

#Menampilkan 5 data teratas
filename_df.head()

#Melihat informasi data
filename_df.info()

#Mengganti tipe data 'Kode Rumah Sakit'
filename_df['Kode Rumah Sakit'] = filename_df['Kode Rumah Sakit'].astype(str)
filename_df['No WhatsApp'] = filename_df['No WhatsApp'].astype(str)

#Melihat informasi data
filename_df.info()

"""# **Data Cleansing**"""

#Mengecek missing value pada data
miss_value = filename_df.isna().mean(axis=0)
miss_value

#Menghapus kolom yang mengandung missing value
filename_drop = filename_df.drop(['Email', 'Kamar'], axis = 1)
filename_drop

"""# **Data Preparation**"""

#Mengecek outlier
data_num = ['Longitude', 'Latitude']
sns.boxplot(filename_drop[data_num])

#Melakukan scaling pada data Longitude dan Latitude
mean = np.mean(filename_drop[data_num])
std = np.std(filename_drop[data_num])

scaled = tf.divide(tf.subtract(filename_drop[data_num], mean), std)
scaled = pd.DataFrame(scaled)
scaled.columns = ['Longitude', 'Latitude']
scaled

#Menghapus kolom yang mengandung Longitude dan Latitude
filename = filename_drop.drop(['Longitude', 'Latitude'], axis = 1)
filename

#Menggabungkan data hasil scaling kedalam filename
filename[scaled.columns] = scaled[scaled.columns]
filename.head()

#Mengurutkan data berdasarkan Longitude, Latitude, Tipe, dan BPJS
data_hosp = filename.sort_values(by=['Longitude', 'Latitude', 'Tipe', 'BPJS'], ascending = False)
data_hosp.head()

"""# **Build Model**

**Mencari nilai K optimal**
"""

#Model dengan inisiasi K=3

#coordinates = filename_drop[['Longitude', 'Latitude']]
X = KMeans(n_clusters = 3)
X.fit(scaled)
#X.fit(coordinates)

#Label cluster
y = X.labels_
y

#Analisis dengan menggunakan metode Elbow

distortions = []
K = range(1,10)
for k in K:
  X = KMeans(n_clusters = k)
  X.fit(scaled)
  #X.fit(coordinates)
  inertia_score = X.inertia_
  distortions.append(inertia_score)

#Plot Elbow untuk setiap cluster
plt.plot(K, distortions, marker = 'o')
plt.xlabel('k')
plt.ylabel('distortions')
plt.title('The Optimal k')
plt.show()

#Analisis dengan menggunakan metode Silhouette
silhouette = []

K = range(2, 10)
for k in K:
  X = KMeans(n_clusters = k, init = 'k-means++')
  X.fit(scaled)
  #X.fit(coordinates)
  labels = X.labels_
  silhouette.append(metrics.silhouette_score(scaled, labels, metric = 'euclidean'))

#Melihat silhouette score dari setiap cluster
sil_score = pd.DataFrame({'Cluster' : K, 'Score' : silhouette})
sil_score

#Model final dengan K-optimal=6
y = KMeans(n_clusters = 6, init = 'k-means++')
y.fit(scaled)
labels = y.labels_
print('k = 6', 'silhouette_score', metrics.silhouette_score(scaled, labels, metric = 'euclidean'))

#Label cluster
labels

#Memberi label cluster untuk setiap data
filename['Cluster'] = y.predict(filename[['Longitude', 'Latitude']])
filename

filename.drop_duplicates(subset=["Nama Rumah Sakit"])[["Nama Rumah Sakit", "Cluster"]].sort_values("Cluster")

#Mengurutkan data yang telah diberi label berdasarkan Longitude, Latitude, Tipe, dan BPJS
data_hosp = filename.sort_values(by=['Longitude', 'Latitude', 'Tipe', 'BPJS'], ascending = False)
data_hosp.head()

"""# **Test Model**"""

def recommended_hospitals(filename, Longitude, Latitude):
  #mengubah inputan mejadi dataframe
  df_input = pd.DataFrame({"Longitude": [Longitude], "Latitude": [Latitude]})

  #fiture scaling
  scaled = pd.DataFrame(tf.divide(tf.subtract(df_input, mean), std))

  #Prediksi cluster
  Cluster = y.predict(tf.reshape(scaled, (1, -1)))[0]
  data_cluster = filename[filename['Cluster']==Cluster].copy()
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
  return data_cluster

Longitude = 110.85664241141414
Latitude = -7.5523142693779475
recommend_hosp = recommended_hospitals(data_hosp, Longitude, Latitude)
recommend_hosp

Longitude = 110.82716015648799
Latitude = -7.554622333033178
recommend_hosp = recommended_hospitals(data_hosp, Longitude, Latitude)
recommend_hosp

Longitude = 110.78881859471946
Latitude = -7.560228238808601
recommend_hosp = recommended_hospitals(data_hosp, Longitude, Latitude)
recommend_hosp

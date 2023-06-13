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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import haversine_distances

#Load Dataset
filename_df = pd.read_csv("/content/Data Rumah Sakit di Surakarta Fiks.csv")

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
filename_drop.head()

"""# **Data Preparation**"""

#Mengecek outlier
data_num = ['Longitude', 'Latitude']
sns.boxplot(filename_drop[data_num])

#Melakukan standarisasi dan scaling pada Longitude dan Latitude
filename_num = filename_drop[data_num]
sc = StandardScaler()
sc.fit(filename_num)
data_scaled = pd.DataFrame(sc.transform(filename_num))
data_scaled.columns = ['Longitude', 'Latitude']
data_scaled

#Menghapus kolom yang mengandung Longitude dan Latitude
filename = filename_drop.drop(['Longitude', 'Latitude'], axis = 1)
filename

#Menggabungkan data hasil scaling kedalam filename
filename[data_scaled.columns] = data_scaled[data_scaled.columns]
filename.head()

#Mengurutkan data berdasarkan Longitude, Latitude, Tipe, dan BPJS
data_hosp = filename.sort_values(by=['Longitude', 'Latitude', 'Tipe', 'BPJS'], ascending = False)
data_hosp.head()

"""# **Build Model**

**Mencari nilai K optimal**
"""

#Model dengan inisiasi K=3
X = KMeans(n_clusters = 3)
X.fit(data_scaled)

#Label cluster
y = X.labels_
y

#Analisis dengan menggunakan metode Elbow

distortions = []
K = range(1,10)
for k in K:
  X = KMeans(n_clusters = k)
  X.fit(data_scaled)
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
  X.fit(data_scaled)
  labels = X.labels_
  silhouette.append(metrics.silhouette_score(data_scaled, labels, metric = 'euclidean'))

#Melihat silhouette score dari setiap cluster
sil_score = pd.DataFrame({'Cluster' : K, 'Score' : silhouette})
sil_score

#Model final dengan K-optimal=6
#coordinates = filename_drop[['Longitude', 'Latitude']]

y = KMeans(n_clusters = 6, init = 'k-means++')
y.fit(data_scaled)
labels = y.labels_
print('k = 6', 'silhouette_score', metrics.silhouette_score(data_scaled, labels, metric = 'euclidean'))

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
  data_scaled = pd.DataFrame(sc.transform(df_input))

  #Prediksi cluster
  Cluster = y.predict(tf.reshape(data_scaled.values, (1, -1)))[0]
  data_cluster = filename[filename['Cluster']==Cluster].copy()
  print(Cluster)

  #Menghitung jarak user dengan rumah sakit
  data_cluster['Hospital Distance'] = data_cluster.apply(lambda x: haversine_distances([[x.Longitude, x.Latitude]], data_scaled.values), axis = 1)

  #Mengurutkan hasil rekomendasi
  col = ['Hospital Distance']
  data_cluster.sort_values(col, ascending = [True]*len(col), inplace = True)

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

Longitude = 110.80649380745398
Latitude = -7.554570044566307
recommend_hosp = recommended_hospitals(data_hosp, Longitude, Latitude)
recommend_hosp
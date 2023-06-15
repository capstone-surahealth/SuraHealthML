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
filename_df = pd.read_csv("/content/Data Rumah Sakit di Surakarta.csv")

#Show top 5 data
filename_df.head()

#View data information
filename_df.info()

#Replace 'Kode Rumah Sakit' and 'No WhatsApp' data types
filename_df['Kode Rumah Sakit'] = filename_df['Kode Rumah Sakit'].astype(str)
filename_df['No WhatsApp'] = filename_df['No WhatsApp'].astype(str)

#View data information
filename_df.info()

"""# **Data Cleansing**"""

#Checking for missing values in the data
miss_value = filename_df.isna().mean(axis=0)
miss_value

#Drop columns that contain missing values
filename_drop = filename_df.drop(['Email', 'Kamar'], axis = 1)
filename_drop

"""# **Data Preparation**"""

#Checking for outliers
data_num = ['Longitude', 'Latitude']
sns.boxplot(filename_drop[data_num])

#Standardizing and scaling Longitude and Latitude
filename_num = filename_drop[data_num]
sc = StandardScaler()
sc.fit(filename_num)
data_scaled = pd.DataFrame(sc.transform(filename_num))
data_scaled.columns = ['Longitude', 'Latitude']
data_scaled

#Drop columns containing Longitude and Latitude
filename = filename_drop.drop(['Longitude', 'Latitude'], axis = 1)
filename

#Merge the scaled data into a filename
filename[data_scaled.columns] = data_scaled[data_scaled.columns]
filename.head()

#Sort data by Longitude, Latitude, Type, and BPJS
data_hosp = filename.sort_values(by=['Longitude', 'Latitude', 'Tipe', 'BPJS'], ascending = False)
data_hosp.head()

"""# **Build Model**

**Find the optimal K value**
"""

#Model with initiation K=3
X = KMeans(n_clusters = 3)
X.fit(data_scaled)

#Label cluster
y = X.labels_
y

#Analysis using the Elbow method
distortions = []
K = range(1,10)
for k in K:
  X = KMeans(n_clusters = k)
  X.fit(data_scaled)
  inertia_score = X.inertia_
  distortions.append(inertia_score)

#Elbow plots for each cluster
plt.plot(K, distortions, marker = 'o')
plt.xlabel('k')
plt.ylabel('distortions')
plt.title('The Optimal k')
plt.show()

#Analysis using the Silhouette method
silhouette = []

K = range(2, 10)
for k in K:
  X = KMeans(n_clusters = k, init = 'k-means++')
  X.fit(data_scaled)
  labels = X.labels_
  silhouette.append(metrics.silhouette_score(data_scaled, labels, metric = 'euclidean'))

#View the silhouette score of each cluster
sil_score = pd.DataFrame({'Cluster' : K, 'Score' : silhouette})
sil_score

#Final model with K-optimal=6
y = KMeans(n_clusters = 6, init = 'k-means++')
y.fit(data_scaled)
labels = y.labels_
print('k = 6', 'silhouette_score', metrics.silhouette_score(data_scaled, labels, metric = 'euclidean'))

#Label cluster
labels

#Labeling each data cluster
filename['Cluster'] = y.predict(filename[['Longitude', 'Latitude']])
filename

#View the cluster results of each hospital
filename.drop_duplicates(subset=["Nama Rumah Sakit"])[["Nama Rumah Sakit", "Cluster"]].sort_values("Cluster")

#Sorting data that has been labeled based on Longitude, Latitude, Type, and BPJS
data_hosp = filename.sort_values(by=['Longitude', 'Latitude', 'Tipe', 'BPJS'], ascending = False)
data_hosp.head()

"""# **Test Model**"""

def recommended_hospitals(filename, Longitude, Latitude):
  #Turning the input into a dataframe
  df_input = pd.DataFrame({"Longitude": [Longitude], "Latitude": [Latitude]})

  #Fiture scaling
  data_scaled = pd.DataFrame(sc.transform(df_input))

  #Cluster prediction
  Cluster = y.predict(tf.reshape(data_scaled, (1, -1)))[0]
  data_cluster = filename[filename['Cluster']==Cluster].copy()
  print(Cluster)
  
  #Fiture inverse
  data_cluster.reset_index(inplace = True, drop = True)
  tmp = data_cluster.apply(lambda x: sc.inverse_transform(np.array([[x.Longitude, x.Latitude]])), axis = 1)
  tmp_long = [tmp[x][0][0] for x in range(len(tmp))];
  tmp_lat = [tmp[x][0][1] for x in range(len(tmp))];
  data_cluster.Longitude = tmp_long; data_cluster.Latitude = tmp_lat;

  #Calculating the distance between the user and the hospital
  data_cluster['Hospital Distance'] = data_cluster.apply(lambda x: haversine_distances([[x.Longitude, x.Latitude]], df_input.values), axis = 1)

  #Sort recommendation results
  col_name = ['Hospital Distance']
  data_cluster.sort_values(col_name, ascending = [True]*len(col_name), inplace = True)

  #Select the number N of recommendations
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
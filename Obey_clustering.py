# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 15:44:05 2021

@author: Garduno
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.tree import export_graphviz
#import pydot
import plotly.express as px
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.decomposition import PCA
import seaborn as sns


def obey(data):
    if data == "Insufficient_Weight":
        return 0
    elif data == "Normal_Weight":
        return 1
    elif data == "Overweight_Level_I":
        return 2
    elif data == "Overweight_Level_II":
        return 3
    elif data == "Obesity_Type_I":
        return 4
    elif data == "Obesity_Type_II":
        return 5
    elif data == "Obesity_Type_III":
        return 6

def Preprocessing(data):
    if data == word:
        return 1
    else:
        return 0

def Normalize(data):
    norm = (data-data.min())/(data.max()-data.min())
    return norm

def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KM:
    def __init__(self, K=7, max_iters=300,steps = False):
        self.K = K
        self.max_iters = max_iters
        self.steps = steps
        #list of list
        self.clusters = [[] for _ in range(self.K)] 
        self.centroids = []
        
    def predict(self,X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        
        #Initialize centroids
        random_sample = np.random.choice(self.n_samples,self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample]
        #Optimization
        for _ in range(self.max_iters):
            #update clusters
            self.clusters = self._create_clusters(self.centroids)
            if self.steps:
                self.plot()
            #update centroids
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            if self.steps:
                self.plot()
            #check if converged
            if self._is_converged(centroids_old, self.centroids):
                break    
        #return cluster labels
        return self._get_cluster_labels(self.clusters)
    
    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for samples_idx in cluster:
                labels[samples_idx] = cluster_idx
        return labels
        
    def _create_clusters(self, centroids):
        clusters = [[]for _ in range(self.K)]
        for idx,sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample,centroids)
            clusters[centroid_idx].append(idx)
        return clusters
    def _closest_centroid(self,sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest = np.argmin(distances)
        return closest
    def _get_centroids(self,clusters):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis = 0)
            centroids[cluster_idx] = cluster_mean
        return centroids
    def _is_converged(self, centroids_old, centroids):
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0
    
    def plot(self,xplt,yplt,labels,save = False,path = None):
        fig, ax = plt.subplots(figsize=(10,10))
        color_theme = np.array(["blue","red","green","black","orange","cyan","magenta"])
        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(x = point[4],y = point[3],c = color_theme[i], s = 50)       
        if save == True:
            plt.savefig(path)
        plt.show()
        
            
        
    


df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
df["FCVC"] = df["FCVC"].apply(np.ceil)
columns_data = ["Gender-Fem","Gender-Male","Age","Height","Weight",
                "History-Yes","History-No","FAVC-Yes","FAVC-No",
               "FCVC-1","FCVC-2","FCVC-3",
               "NCP","CAEC-No","CAEC-Sometimes",
                "CAEC-Frequently","CAEC-Always","SMOKE-Yes","SMOKE-No",
               "CH2O","SCC-Yes","SCC-No","FAF","TUE","CALC-No",
                "CALC-Sometimes","CALC-Frequently","CALC-Always",
               "MTRANS-Auto","MTRANS-Moto","MTRANS-Bike","MTRANS-Public",
               "MTRANS-Walking","NObeyesdad"]
df_data = pd.DataFrame(columns = columns_data)


pd.set_option('display.max_columns', 500)
word =  "Female"
df_data["Gender-Fem"] = df["Gender"].apply(Preprocessing)
word =  "Male"
df_data["Gender-Male"] = df["Gender"].apply(Preprocessing)
df_data["Age"] = Normalize(df["Age"])
df_data["Height"] = Normalize(df["Height"])
df_data["Weight"] = Normalize(df["Weight"])
word =  "yes"
df_data["History-Yes"] = df["family_history_with_overweight"].apply(Preprocessing)
word =  "no"
df_data["History-No"] = df["family_history_with_overweight"].apply(Preprocessing)
word =  "yes"
df_data["FAVC-Yes"] = df["FAVC"].apply(Preprocessing)
word =  "no"
df_data["FAVC-No"] = df["FAVC"].apply(Preprocessing)
df["FCVC"] = df["FCVC"].round()
word = 1
df_data["FCVC-1"] = df["FCVC"].apply(Preprocessing)
word = 2
df_data["FCVC-2"] = df["FCVC"].apply(Preprocessing)
word = 3
df_data["FCVC-3"] = df["FCVC"].apply(Preprocessing)
df_data["NCP"] = Normalize(df["NCP"])
word = "No"
df_data["CAEC-No"] = df["CAEC"].apply(Preprocessing)
word = "Sometimes"
df_data["CAEC-Sometimes"] = df["CAEC"].apply(Preprocessing)
word = "Frequently"
df_data["CAEC-Frequently"] = df["CAEC"].apply(Preprocessing)
word = "Always"
df_data["CAEC-Always"] = df["CAEC"].apply(Preprocessing)
word = "yes"
df_data["SMOKE-Yes"] = df["SMOKE"].apply(Preprocessing)
word = "no"
df_data["SMOKE-No"] = df["SMOKE"].apply(Preprocessing)
df_data["CH2O"] = Normalize(df["CH2O"])
word = "yes"
df_data["SCC-Yes"] = df["SCC"].apply(Preprocessing)
word = "no"
df_data["SCC-No"] = df["SCC"].apply(Preprocessing)
df_data["FAF"] = Normalize(df["FAF"])
df_data["TUE"] = Normalize(df["TUE"])
word = "No"
df_data["CALC-No"] = df["CALC"].apply(Preprocessing)
word = "Sometimes"
df_data["CALC-Sometimes"] = df["CALC"].apply(Preprocessing)
word = "Frequently"
df_data["CALC-Frequently"] = df["CALC"].apply(Preprocessing)
word = "Always"
df_data["CALC-Always"] = df["CALC"].apply(Preprocessing)
word = "Automobile"
df_data["MTRANS-Auto"] = df["MTRANS"].apply(Preprocessing)
word = "Motorbike"
df_data["MTRANS-Moto"] = df["MTRANS"].apply(Preprocessing)
word = "Bike"
df_data["MTRANS-Bike"] = df["MTRANS"].apply(Preprocessing)
word = "Public_Transportation"
df_data["MTRANS-Public"] = df["MTRANS"].apply(Preprocessing)
word = "Walking"
df_data["MTRANS-Walking"] = df["MTRANS"].apply(Preprocessing)
df_data["NObeyesdad"] = df["NObeyesdad"]
df_data["Original"] = df["NObeyesdad"].apply(obey)

Names = ["Gender-Fem","Gender-Male","Age","Height","Weight",
                "History-Yes","History-No","FAVC-Yes","FAVC-No",
               "FCVC-1","FCVC-2","FCVC-3",
               "NCP","CAEC-No","CAEC-Sometimes",
                "CAEC-Frequently","CAEC-Always","SMOKE-Yes","SMOKE-No",
               "CH2O","SCC-Yes","SCC-No","FAF","TUE","CALC-No",
                "CALC-Sometimes","CALC-Frequently","CALC-Always",
               "MTRANS-Auto","MTRANS-Moto","MTRANS-Bike","MTRANS-Public",
               "MTRANS-Walking"]
X = df_data[Names].to_numpy()
k = 7
cluster = KMeans(n_clusters = k,
                 init="k-means++",
                 n_init=50,
                 max_iter=500,
                 random_state=54)
prediction = cluster.fit_predict(X)
df_data["Labels"] = prediction

fig = plt.figure(figsize = (10,10))
plot1 = fig.add_subplot(2,2,1)
plot1.set_xlabel("Weight", fontsize = 15)
plot1.set_ylabel("Height", fontsize = 15)
plot1.set_title("Cluster", fontsize = 20)
color_theme = np.array(["blue","red","green","black","orange","cyan","magenta"])
plot1.scatter(x = df.Weight,y = df.Height, c = color_theme[df_data.Labels], s = 50)

plot2 = fig.add_subplot(2,2,2)
plot2.set_xlabel("Weight", fontsize = 15)
plot2.set_ylabel("Height", fontsize = 15)
plot2.set_title("Original", fontsize = 20)
plot2.scatter(x = df.Weight,y = df.Height, c = color_theme[df_data.Original], s = 50)

plt.savefig("Cluster_Original.png")

fig2 = px.scatter_matrix(df_data,dimensions = Names,color = "Labels")
fig2.update_traces(diagonal_visible=False)
fig2.update_layout(title="Obey analysis",
                  dragmode='select',
                  width=4000,
                  height=4000,
                  hovermode='closest')
fig2.write_image("Scatter.png")



np.random.seed(42)

hand_cluster = KM(K=7,max_iters = 150, steps=False)
y_hand = hand_cluster.predict(X)
hand_cluster.plot(df.Weight,df.Height,y_hand,save = True,path = "By_Hand_Cluster.png")


#Rand index
rand_idx = adjusted_rand_score(df_data.Original, df_data.Labels)

print("Rand index " + str(rand_idx))

pca = PCA(n_components=2,random_state = 42)
X_pca = pca.fit_transform(X)
cluster2 = KMeans(n_clusters = k,
                 init="k-means++",
                 n_init=50,
                 max_iter=500,
                 random_state=54)
prediction2 = cluster.fit_predict(X_pca)
df_data["Labels_pca"] = prediction2

fig3 = plt.figure(figsize = (10,10))

pcadf = pd.DataFrame(X_pca,
    columns=["component_1", "component_2"],
)

pcadf["predicted_cluster"] = df_data["Labels_pca"]
pcadf["true_label"] = df_data["NObeyesdad"]

scat = sns.scatterplot(
    "component_1",
    "component_2",
    s=50,
    data=pcadf,
    hue="predicted_cluster",
    style="true_label",
    palette="Set2",
)

scat.set_title(
    "Clustering PCA"
)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

plt.autoscale()
plt.savefig("Cluster_pca.png", bbox_inches = "tight")

#Rand index PCA
rand_idx_pca = adjusted_rand_score(df_data.Original, df_data.Labels_pca)

print("Rand index PCA " + str(rand_idx_pca))

import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from keras.models import Model
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from skfuzzy.cluster import cmeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from ELE690_functions import *




# Load DataFrame from the CSV file
df = pd.read_csv('patients_and_paths.csv')

# make patient labels and create the dataset
patient_id_mapping = {id_: idx for idx, id_ in enumerate(df['patient'].unique())}
df['patient_label'] = df['patient'].map(patient_id_mapping)
dataset = create_dataset(df, batch_size=32)

# initialize the VGG16 model without the classifying layer
model = VGG16()
model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

# create the feature space
predictions = model.predict(dataset)

print(f'Predictions shape: {predictions.shape}')

# reduce dimensionality linearly with PCA
pca = PCA(n_components=2, random_state=22)
pca.fit(predictions)
predictions_reduced = pca.transform(predictions)

# kmeans clustering
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(predictions)
cluster_labels_kmean = kmeans.labels_
df['cluster_kmean'] = cluster_labels_kmean

# PCA plot
'''
plt.scatter(predictions_reduced[:, 0], predictions_reduced[:, 1], c=cluster_labels_kmean, cmap='viridis')
plt.title('Kmeans clusters Visualized with PCA')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster Label')
plt.show()
'''

# Agglomerate clustering
'''
agg_clustering = AgglomerativeClustering(n_clusters=5) # prøv med unspecified n_clusters, må setta distance threshold
cluster_labels_agg = agg_clustering.fit_predict(predictions)
print(f'Cluster labels from Agglomerative Clustering: {cluster_labels_agg}')
df['cluster_agg'] = cluster_labels_agg

plt.scatter(predictions_reduced[:, 0], predictions_reduced[:, 1], c=cluster_labels_agg, cmap='viridis')
plt.title('Agglomerate clusters Visualized with PCA')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster Label')
plt.show()
'''

# t-sne
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)  
reduced_data_sne = tsne.fit_transform(predictions)

plt.figure(figsize=(10, 7))
plt.scatter(reduced_data_sne[:, 0], reduced_data_sne[:, 1], c=cluster_labels_kmean, cmap='viridis') 
plt.title('K-Means clusters viualized with t-sne')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(label='Cluster Label')
plt.show()


patient_clusters = df.groupby('patient')['cluster_kmean'].apply(list).to_dict()
print(f'patients in dict: {[*patient_clusters.keys()]}')
ex_patient = list(patient_clusters.keys())[0]
print(f'clusters for single patient: {patient_clusters[ex_patient]}')

# assign each patient to a cluster

patient_likely_cluster = {}
for key, value in patient_clusters.items():
    patient_likely_cluster[key] = max(set(value), key=value.count)

print(f'patients most common cluster: {patient_likely_cluster}')


# create pandas dataframe
df_clusters = pd.DataFrame(list(patient_likely_cluster.items()), columns=['PM', 'cluster'])
df_clusters['infarct_size_manual'] = None
df_clusters['all_cause_mortality'] = None
df_clusters['gender'] = None
df_clusters['age'] = None
df_clusters['cholesterol'] = None
df_clusters['HDL_cholesterol'] = None


# load the excel files
variables_PM_study = pd.read_excel('/home/prosjekt5/BMDLab/data/mri/PM/Variables PM-study.xlsx', index_col=0)
variables_PM_study.index = variables_PM_study.index.astype(str)
variables_PM_study.index = variables_PM_study.index.map(lambda x: f"PM{int(x):03}")
PM_materiale = pd.read_excel('/home/prosjekt5/BMDLab/data/mri/PM/PM-materiale.xlsx', index_col=0)
PM_materiale.index = PM_materiale.index.astype(str)
PM_materiale.index = PM_materiale.index.map(lambda x: f"PM{int(x):03}")

for _, row in variables_PM_study.iterrows():
    patient_number = row.name
    #print(f'patient_number: {patient_number}')
    if patient_number in df_clusters['PM'].values:
        infarct_size_manual = row['Infarct size manual']
        all_cause_mortality = row['All cause mortality']
        df_clusters.loc[df_clusters['PM'] == patient_number, 'infarct_size_manual'] = infarct_size_manual
        df_clusters.loc[df_clusters['PM'] == patient_number, 'all_cause_mortality'] = all_cause_mortality

for _, row in PM_materiale.iterrows():
    patient_number = row.name
    if patient_number in df_clusters['PM'].values:
        gender = row['Gender']
        age = row['Age_at_inlusion']
        cholesterol = row['Cholesterol_total']
        HDL_cholesterol = row['HDL_Cholesterol']
        df_clusters.loc[df_clusters['PM'] == patient_number, 'gender'] = gender
        df_clusters.loc[df_clusters['PM'] == patient_number, 'age'] = age
        df_clusters.loc[df_clusters['PM'] == patient_number, 'cholesterol'] = cholesterol
        df_clusters.loc[df_clusters['PM'] == patient_number, 'HDL_cholesterol'] = HDL_cholesterol

# see the distribution of the different factors in clusters
print('infarct_size_manual:')
cluster_summary = df_clusters.groupby('cluster')['infarct_size_manual'].describe()
print(cluster_summary)
print('all_cause_mortality')
cluster_summary = df_clusters.groupby('cluster')['all_cause_mortality'].describe()
print(cluster_summary)
print('gender')
cluster_summary = df_clusters.groupby('cluster')['gender'].describe()
print(cluster_summary)
print('age')
cluster_summary = df_clusters.groupby('cluster')['age'].describe()
print(cluster_summary)
print('Cholesterol total')
cluster_summary = df_clusters.groupby('cluster')['cholesterol'].describe()
print(cluster_summary)
print('HDL_Cholesterol')
cluster_summary = df_clusters.groupby('cluster')['HDL_cholesterol'].describe()
print(cluster_summary)

# box plots
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_clusters, x='cluster', y='infarct_size_manual')
plt.title('infarct size distribution by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Infarct size manual')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_clusters, x='cluster', y='all_cause_mortality')
plt.title('all cause mortality distribution by Cluster')
plt.xlabel('Cluster')
plt.ylabel('All cause mortality')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_clusters, x='cluster', y='gender')
plt.title('Gender distribution by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Gender')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_clusters, x='cluster', y='age')
plt.title('Age distribution by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Age')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_clusters, x='cluster', y='cholesterol')
plt.title('Cholesterol distribution by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Cholesterol')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_clusters, x='cluster', y='HDL_cholesterol')
plt.title('HDL_Cholesterol distribution by Cluster')
plt.xlabel('Cluster')
plt.ylabel('HDL_Cholesterol')
plt.show()


# scatter plots
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_clusters, x='infarct_size_manual', y='age', hue='cluster', palette='viridis')
plt.title('Infarct size manual vs. age')
plt.xlabel('Infarct size manual')
plt.ylabel('Age')
plt.legend(title='Cluster')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_clusters, x='cholesterol', y='HDL_cholesterol', hue='cluster', palette='viridis')
plt.title('Cholesterol vs. HDL_cholesterol')
plt.xlabel('Cholesterol total')
plt.ylabel('HDL_Cholesterol')
plt.legend(title='Cluster')
plt.show()

# save dataframe and make the silhouette score
df_clusters.to_csv('vgg16.csv', index=False)
sil_score = silhouette_score(predictions, cluster_labels_kmean)
print(f"Silhouette Score: {sil_score}")




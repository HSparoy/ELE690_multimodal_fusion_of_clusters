#########################################################
# only for use with gorina GPU's
'''
import socket

if socket.gethostname() == 'go6' or socket.gethostname() == 'go4':
    print('GPU settings for manual reservation')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # GPU number
'''
#########################################################

import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from keras.models import Model
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from skfuzzy.cluster import cmeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from ELE690_functions import *


# Load DataFrame from the CSV file
df = pd.read_csv('patients_and_paths.csv') # 'patient' and 'file'
#dataset = create_dataset(df, batch_size=32)

# make patient labels for the model
patient_id_mapping = {id_: idx for idx, id_ in enumerate(df['patient'].unique())}
df['patient_label'] = df['patient'].map(patient_id_mapping)
print(df['patient_label'])
train_df, val_df = train_test_split(df, stratify=df['patient_label'], test_size=0.2, random_state=42)

# initialize the model
num_patients = df['patient_label'].nunique()
input_shape = (224, 224, 1)  
model = create_patient_id_model(input_shape, num_patients)

# create the datasets
train_dataset = create_dataset(train_df, batch_size=32)
valid_dataset = create_dataset(val_df, batch_size=32)
dataset = create_dataset(df, batch_size=32)

# Create ImageDataGenerators for loading and augmenting the images
'''
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_data, x_col='file', y_col='patient_label',
    target_size=(224, 224), class_mode='raw', batch_size=32
)

val_generator = val_datagen.flow_from_dataframe(
    val_data, x_col='file', y_col='patient_label',
    target_size=(224, 224), class_mode='raw', batch_size=32
)
'''

# early stopping
early_stopping = EarlyStopping(monitor="val_loss", mode="auto", verbose=1, patience=5)

# checkpoint
model_name = 'test_model'
model_checkpoint = ModelCheckpoint(f'{model_name}.keras', monitor="val_loss", mode="min", verbose=1, save_best_only=True)

# check datatypes
for x, y in train_dataset.take(1):
    print("Image shape:", x.shape)  # Should be (batch_size, 224, 224, 1)
    print("Label shape:", y.shape) 
print(df['patient_label'].dtype)  # Should be int or int64

# Train the model
history = model.fit(train_dataset, 
                    validation_data=valid_dataset, 
                    epochs=10,
                    verbose=2,
                    callbacks=[early_stopping, model_checkpoint])


# create the feature space for clustering
feature_extraction_model = models.Model(inputs=model.input, outputs=model.layers[-2].output)
features = feature_extraction_model.predict(dataset)
features_before = features
feature_dict = {path: feature for path, feature in zip(df['file'], features)}

# initial clustering, either kmeans or gaussian mixture 
# Choice will also be used for triplet loss, change variable in triplet loss accordingly

# kmeans
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit_predict(features)
cluster_labels_kmean_before = kmeans.labels_

# gmm clustering
'''
n_clusters = 3
gmm = GaussianMixture(n_components=n_clusters, random_state=42)
gmm.fit(features)
cluster_labels_kmean_before = gmm.predict(features)
'''
#########################################
# DeepCluster Loss (simple version)
'''
dataset_single = create_dataset(df, 1)
pseudo_labels = kmeans.fit_predict(features)
images = []
for x, y in dataset_single:  
    images.append(x.numpy())
images = np.array(images)
images = images.squeeze(axis=1)
pseudo_dataset = tf.data.Dataset.from_tensor_slices((images, pseudo_labels)).batch(32)
model.fit(pseudo_dataset, epochs=10)
feature_extraction_model = models.Model(inputs=model.input, outputs=model.layers[-2].output)
features = feature_extraction_model.predict(dataset)
feature_dict = {path: feature for path, feature in zip(df['file'], features)}
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(features)

cluster_labels_kmean = kmeans.labels_
df['cluster_kmean'] = cluster_labels_kmean
'''
##############################################

##############################################
# triplet loss
optimizer = Adam(learning_rate=0.001)
# initialize the cluster model
cluster_model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(128,)),  
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(128, activation=None)
])
# combine the model for triplet loss predictions
combined_model = tf.keras.Sequential([
    feature_extraction_model,  # Pretrained feature extractor
    cluster_model              # Triplet tuning layers
])
# triplet loss for n epochs
for epoch in range(10):  
    # Extract features and generate pseudo-labels
    print(f'Starting deep cluster tuning epoch {epoch}')
    features = combined_model.predict(dataset)
    # kmeans.fit_predict or gmm.fit_predict according to triplet loss method
    pseudo_labels = kmeans.fit_predict(features) 
    ##############################################################
    # generate every triplet and filter the hard ones out
    '''
    # Generate triplets for triplet loss
    triplets = generate_triplets(features, pseudo_labels)
    print(f"Generated {len(triplets)} triplets.")
    # to reduce processing time filter out just the hardest example triplets
    # which provide the most useful gradient updates

    triplets = filter_hard_triplets(triplets) 
    print(f"Filtered {len(triplets)} hard triplets for training.")
    '''
    ###############################################################
    # Create only the hard triplets
    triplets = generate_hard_triplets(features, pseudo_labels)
    print(f"Generated {len(triplets)} hard triplets.")

    # Convert to TensorFlow Dataset
    triplets_tensor = tf.convert_to_tensor(triplets) # unnecesarry, but will get warnings
    triplet_dataset = tf.data.Dataset.from_tensor_slices(triplets_tensor).batch(1)

    # Train the cluster model model
    for batch in triplet_dataset:
        with tf.GradientTape() as tape:
            anchor, positive, negative = batch[:, 0], batch[:, 1], batch[:, 2]
            anchor_embed = cluster_model(anchor, training=True)
            positive_embed = cluster_model(positive, training=True)
            negative_embed = cluster_model(negative, training=True)
            loss = triplet_loss(anchor_embed, positive_embed, negative_embed)
        gradients = tape.gradient(loss, cluster_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, cluster_model.trainable_variables))

    # recombine the feature extractor model with the tuned cluster model
    combined_model = tf.keras.Sequential([
        feature_extraction_model,  
        cluster_model              
])

# create the tuned feature space
feature_extraction_model = models.Model(inputs=model.input, outputs=model.layers[-2].output)
features = feature_extraction_model.predict(dataset)
features = cluster_model.predict(features)

# make tuned clusters either kmeans or gmm
#kmeans after triplet loss
kmeans_after = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_after.fit_predict(features)
cluster_labels_kmean = kmeans_after.labels_

# GMM after triplet loss
'''
gmm_after = GaussianMixture(n_components=n_clusters, random_state=42)
gmm_after.fit(features)
cluster_labels_kmean = gmm_after.predict(features)
#cluster_labels_kmean_before = gmm.predict(features_before)
'''

df['cluster_kmean'] = cluster_labels_kmean
cluster_model.save('deep_cluster_gmm.keras')



##############################################

# save model after tuning
model.save(f'{model_name}_tuned.keras')
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

# t-sne original cluster plot
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)  
reduced_data_sne = tsne.fit_transform(features_before)

plt.figure(figsize=(10, 7))
plt.scatter(reduced_data_sne[:, 0], reduced_data_sne[:, 1], c=cluster_labels_kmean_before, cmap='viridis') 
plt.title('Gaussian mixture model clusters viualized with t-sne')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(label='Cluster Label')
plt.show()

# t-sne cluster plot after triplet loss
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)  
reduced_data_sne = tsne.fit_transform(features)

plt.figure(figsize=(10, 7))
plt.scatter(reduced_data_sne[:, 0], reduced_data_sne[:, 1], c=cluster_labels_kmean, cmap='viridis') 
plt.title('K-Means clusters after triplet loss viualized with t-sne')
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

# extract characteristics
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
df_clusters.to_csv('gmm_kmeans_new_3.csv', index=False)
sil_score = silhouette_score(features, cluster_labels_kmean)
print(f"Silhouette Score: {sil_score}")

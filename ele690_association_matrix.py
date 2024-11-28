
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt


#Load the cvs files with clusters
text_clusters = pd.read_csv("/home/prosjekt/BMDLab/users/ELE690_2024/clustered_patient_reports_combined_tsne.csv")[['Patient_Id', 'Cluster']].rename(columns={'Cluster': 'Cluster_Text'})
image_clusters = pd.read_csv("/home/prosjekt/BMDLab/users/ELE690_2024/gmm_3.csv")[['PM', 'cluster']].rename(columns={'PM': 'Patient_Id','cluster': 'Cluster_Image'})
# Merge the datasets on Patient_Id
merged_df = pd.merge(text_clusters, image_clusters, on="Patient_Id")

# Initialize an empty association matrix
num_patients = merged_df.shape[0]
association_matrix = np.zeros((num_patients, num_patients))

# Compute the weighted similarity between clusters
text_weight = 0.4
image_weight = 0.6

for i in range(num_patients):
    for j in range(num_patients):
        sim_text = 1 - abs(merged_df.loc[i, 'Cluster_Text'] - merged_df.loc[j, 'Cluster_Text'])
        sim_image = 1 - abs(merged_df.loc[i, 'Cluster_Image'] - merged_df.loc[j, 'Cluster_Image'])
        association_matrix[i, j] = text_weight * sim_text + image_weight * sim_image

# Convert similarity matrix to distance matrix
distance_matrix = 1 - association_matrix

# Ensure non-negative values in the distance matrix
# Clamp negative values to 0
distance_matrix = np.maximum(distance_matrix, 0)

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, metric="precomputed", init="random", random_state=42)
reduced_data = tsne.fit_transform(distance_matrix)


# Align clusters based on patient overlap
# Create overlap matrix
text_patient_clusters = merged_df.groupby('Cluster_Text')['Patient_Id'].apply(set)
image_patient_clusters = merged_df.groupby('Cluster_Image')['Patient_Id'].apply(set)

num_text_clusters = len(text_patient_clusters)
num_image_clusters = len(image_patient_clusters)
overlap_matrix = np.zeros((num_text_clusters, num_image_clusters), dtype=int)

for i, text_cluster_patients in text_patient_clusters.items():
    for j, image_cluster_patients in image_patient_clusters.items():
        overlap_matrix[i, j] = len(text_cluster_patients & image_cluster_patients)  # Intersection size

print("Cluster Overlap Matrix (Text vs. Image):\n", overlap_matrix)

# Use Hungarian algorithm to maximize overlap
row_indices, col_indices = linear_sum_assignment(-overlap_matrix)
cluster_mapping = {text_cluster: image_cluster for text_cluster, image_cluster in zip(row_indices, col_indices)}
print("Optimal Cluster Mapping (Text -> Image):", cluster_mapping)

# Map text cluster labels to align with image clusters
merged_df['Cluster_Text_Mapped'] = merged_df['Cluster_Text'].map(cluster_mapping)

# Save the aligned clusters
merged_df.to_csv('aligned_clusters_with_kmeans.csv', index=False)
print("Aligned clusters saved to 'aligned_clusters_with_kmeans.csv'")


#Print information about the assosiation matrix
print(association_matrix)
print("Matrix variance:", np.var(association_matrix))

print("Association Matrix Statistics:")
print("Min:", np.min(association_matrix))
print("Max:", np.max(association_matrix))
print("Mean:", np.mean(association_matrix))
print("Std Dev:", np.std(association_matrix))

print("Text Weight:", text_weight)
print("Image Weight:", image_weight)
print("Example Weighted Similarity:", text_weight * 0.8 + image_weight * 0.2)

print("Association Matrix (Top-5 Rows and Columns):")
print(association_matrix[:5, :5])





# Normalize cluster differences
max_possible_difference = max(
    merged_df['Cluster_Text'].max() - merged_df['Cluster_Text'].min(),
    merged_df['Cluster_Image'].max() - merged_df['Cluster_Image'].min()
)

# Calculate normalized similarity scores
for i in range(num_patients):
    for j in range(num_patients):
        sim_text = 1 - abs(merged_df.loc[i, 'Cluster_Text'] - merged_df.loc[j, 'Cluster_Text']) / max_possible_difference
        sim_image = 1 - abs(merged_df.loc[i, 'Cluster_Image'] - merged_df.loc[j, 'Cluster_Image']) / max_possible_difference
        association_matrix[i, j] = text_weight * sim_text + image_weight * sim_image

# Clamp values to [0, 1]
association_matrix = np.clip(association_matrix, 0, 1)

# Print statistics
print("Association Matrix Statistics:")
print("Min:", np.min(association_matrix))
print("Max:", np.max(association_matrix))
print("Mean:", np.mean(association_matrix))
print("Std Dev:", np.std(association_matrix))



# Visualize the aligned clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(
    x=reduced_data[:, 0],
    y=reduced_data[:, 1],
    hue=merged_df['Cluster_Text_Mapped'],  # Use mapped text clusters
    palette="viridis",
    s=100
)
plt.title('t-SNE + Aligned Clusters (Based on Patient Overlap)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(title='Cluster', loc='best', bbox_to_anchor=(1, 1))
plt.show()


# Prepare the heatmap data as a DataFrame
heatmap_data = pd.DataFrame(
    association_matrix,
    index=merged_df['Patient_Id'],  # Patient IDs for y-axis
    columns=merged_df['Patient_Id']  # Patient IDs for x-axis
)

# Prepare the heatmap data as a DataFrame
heatmap_data = pd.DataFrame(
    association_matrix,
    index=merged_df['Patient_Id'],  # Patient IDs for y-axis
    columns=merged_df['Patient_Id']  # Patient IDs for x-axis
)

# Generate ticks for every n-th patient
num_patients = len(merged_df['Patient_Id'])
ticks = list(range(0, num_patients, 3))  # Indices for every n-th patient
tick_labels = merged_df['Patient_Id'].iloc[ticks]  # Labels for every 5th patient

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    heatmap_data,
    cmap='viridis',  # Color scheme
    xticklabels=ticks,  # Set ticks for x-axis
    yticklabels=ticks,  # Set ticks for y-axis
    cbar_kws={'label': 'Similarity'}  # Add a colorbar label
)

# Set the tick labels for every n-th patient
plt.xticks(ticks, tick_labels, rotation=90, fontsize=8)  # Rotate for better readability
plt.yticks(ticks, tick_labels, fontsize=5)

# Add axis labels and title
plt.title("Co-Association Matrix Heatmap with Patient IDs")
plt.xlabel("Patient ID")
plt.ylabel("Patient ID")

# Adjust layout to fit labels
plt.tight_layout()

# Show the plot
plt.show()





# Group combined clusters by patients
patient_clusters_combined = merged_df.groupby('Patient_Id')['Cluster_Text_Mapped'].apply(list).to_dict()

# Print the clusters for debugging
print(f'Patients in dict: {[*patient_clusters_combined.keys()]}')
ex_patient = list(patient_clusters_combined.keys())[0]
print(f'Clusters for single patient: {patient_clusters_combined[ex_patient]}')

# Assign each patient to their most likely cluster
patient_likely_cluster_combined = {}
for key, value in patient_clusters_combined.items():
    patient_likely_cluster_combined[key] = max(set(value), key=value.count)  # Most frequent cluster

print(f'Patients most common cluster (Combined): {patient_likely_cluster_combined}')

# Create a DataFrame for patient clusters
df_clusters_combined = pd.DataFrame(list(patient_likely_cluster_combined.items()), columns=['Patient_Id', 'Combined_Cluster'])
df_clusters_combined['infarct_size_manual'] = None
df_clusters_combined['all_cause_mortality'] = None
df_clusters_combined['gender'] = None
df_clusters_combined['age'] = None
df_clusters_combined['cholesterol'] = None
df_clusters_combined['HDL_cholesterol'] = None

# Load clinical data from Excel files
variables_PM_study = pd.read_excel('/home/prosjekt5/BMDLab/data/mri/PM/Variables PM-study.xlsx', index_col=0)
variables_PM_study.index = variables_PM_study.index.astype(str)
variables_PM_study.index = variables_PM_study.index.map(lambda x: f"PM{int(x):03}")
PM_materiale = pd.read_excel('/home/prosjekt5/BMDLab/data/mri/PM/PM-materiale.xlsx', index_col=0)
PM_materiale.index = PM_materiale.index.astype(str)
PM_materiale.index = PM_materiale.index.map(lambda x: f"PM{int(x):03}")

# Map clinical data to combined clusters
for _, row in variables_PM_study.iterrows():
    patient_number = row.name
    if patient_number in df_clusters_combined['Patient_Id'].values:
        infarct_size_manual = row['Infarct size manual']
        all_cause_mortality = row['All cause mortality']
        df_clusters_combined.loc[df_clusters_combined['Patient_Id'] == patient_number, 'infarct_size_manual'] = infarct_size_manual
        df_clusters_combined.loc[df_clusters_combined['Patient_Id'] == patient_number, 'all_cause_mortality'] = all_cause_mortality

for _, row in PM_materiale.iterrows():
    patient_number = row.name
    if patient_number in df_clusters_combined['Patient_Id'].values:
        gender = row['Gender']
        age = row['Age_at_inlusion']
        cholesterol = row['Cholesterol_total']
        HDL_cholesterol = row['HDL_Cholesterol']
        df_clusters_combined.loc[df_clusters_combined['Patient_Id'] == patient_number, 'gender'] = gender
        df_clusters_combined.loc[df_clusters_combined['Patient_Id'] == patient_number, 'age'] = age
        df_clusters_combined.loc[df_clusters_combined['Patient_Id'] == patient_number, 'cholesterol'] = cholesterol
        df_clusters_combined.loc[df_clusters_combined['Patient_Id'] == patient_number, 'HDL_cholesterol'] = HDL_cholesterol

# Analyze cluster distributions
print('infarct_size_manual:')
cluster_summary = df_clusters_combined.groupby('Combined_Cluster')['infarct_size_manual'].describe()
print(cluster_summary)
print('all_cause_mortality:')
cluster_summary = df_clusters_combined.groupby('Combined_Cluster')['all_cause_mortality'].describe()
print(cluster_summary)
print('gender:')
cluster_summary = df_clusters_combined.groupby('Combined_Cluster')['gender'].describe()
print(cluster_summary)
print('age:')
cluster_summary = df_clusters_combined.groupby('Combined_Cluster')['age'].describe()
print(cluster_summary)
print('Cholesterol total:')
cluster_summary = df_clusters_combined.groupby('Combined_Cluster')['cholesterol'].describe()
print(cluster_summary)
print('HDL_Cholesterol:')
cluster_summary = df_clusters_combined.groupby('Combined_Cluster')['HDL_cholesterol'].describe()
print(cluster_summary)

# Visualize distributions with box plots
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_clusters_combined, x='Combined_Cluster', y='infarct_size_manual')
plt.title('Infarct size distribution by Combined Cluster')
plt.xlabel('Combined Cluster')
plt.ylabel('Infarct size manual')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_clusters_combined, x='Combined_Cluster', y='all_cause_mortality')
plt.title('All cause mortality distribution by Combined Cluster')
plt.xlabel('Combined Cluster')
plt.ylabel('All cause mortality')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_clusters_combined, x='Combined_Cluster', y='age')
plt.title('Age distribution by Combined Cluster')
plt.xlabel('Combined Cluster')
plt.ylabel('Age')
plt.show()

# Scatter plots for comparisons
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_clusters_combined, x='infarct_size_manual', y='age', hue='Combined_Cluster', palette='viridis')
plt.title('Infarct size manual vs. Age by Combined Cluster')
plt.xlabel('Infarct size manual')
plt.ylabel('Age')
plt.legend(title='Combined Cluster')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_clusters_combined, x='cholesterol', y='HDL_cholesterol', hue='Combined_Cluster', palette='viridis')
plt.title('Cholesterol vs. HDL Cholesterol by Combined Cluster')
plt.xlabel('Cholesterol')
plt.ylabel('HDL Cholesterol')
plt.legend(title='Combined Cluster')
plt.show()


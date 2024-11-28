import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk

# Download stopwords if not already downloaded
#nltk.download('stopwords')

# Load processed data from CSV
csv_path = 'clinical_notes_MRI_projcet.csv'
try:
    patient_report_df = pd.read_csv(csv_path)
except FileNotFoundError:
    print("Processed data not found. Generate the CSV file first.")
    exit()

# Initialize NorBERT model and tokenizer for embeddings
tokenizer = AutoTokenizer.from_pretrained("NbAiLab/nb-bert-base")
model = AutoModel.from_pretrained("NbAiLab/nb-bert-base")

# Define Norwegian stopwords and add specific custom stopwords
norwegian_stopwords = set(stopwords.words('norwegian'))
custom_stopwords = norwegian_stopwords.union({
    "rapport", "pasient", "undersøkelse", "med", "og", "av", "i", 
    "til", "for", "på", "fra", "som", "er", "det", "den", "Rogaland", "cor"
})

# Function to remove stopwords and short/meaningless words from text
def remove_stopwords(text):
    words = text.split()
    filtered_words = [
        word for word in words 
        if word.lower() not in custom_stopwords and len(word) > 2  # Remove short words
    ]
    return ' '.join(filtered_words)

# Function to get embeddings for a specific text
def get_embeddings(text_data):
    inputs = tokenizer(text_data, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# Number of clusters fixed at choosen number
num_clusters = 3

# List of fields to process separately
fields = ['Clinical_information', 'Desired_examination_treatment', 'Section_report', 'MRI_of_cor', 'Diagnosis']

# Specify the fields to skip
skip_fields = [''] 

# Exclude the fields in skip_fields from the list
fields_to_plot = [field for field in fields if field not in skip_fields]

# Dynamically set up side-by-side plot for the remaining fields
num_fields = len(fields_to_plot)
fig, axes = plt.subplots(1, num_fields, figsize=(5 * num_fields, 6))
fig.suptitle('K-Means Clusters for Each Field (t-SNE-reduced to 2D)', fontsize=16)

# Loop through each field to create clusters and assign descriptions
for i, field in enumerate(fields_to_plot):  # Use fields_to_plot to exclude the skipped field
    print(f"\nProcessing clustering for field: {field}")

    # Ensure NaN values are filled with empty strings
    patient_report_df[field] = patient_report_df[field].fillna('')

    # Filter out rows with empty text
    field_data = patient_report_df[patient_report_df[field].str.strip() != '']

    # Generate embeddings for each entry in the field
    embeddings = []
    for text in field_data[field]:
        embedding = get_embeddings(text)
        embeddings.append(embedding)

    # Stack all embeddings into a single tensor for clustering
    embeddings_tensor = torch.cat(embeddings).numpy()

    # Perform K-Means clustering with 5 clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings_tensor)
    field_data['Cluster'] = cluster_labels

    # Analyze each cluster to assign descriptive names
    cluster_names = {}
    colors = plt.cm.get_cmap('viridis', num_clusters)  # Get a color map with a color for each cluster
    for cluster in range(num_clusters):
        cluster_data = field_data[field_data['Cluster'] == cluster]
        
        # Combine all text in the cluster and remove stopwords
        all_text = ' '.join(cluster_data[field].dropna().tolist())
        all_text_no_stopwords = remove_stopwords(all_text)
        
        # Find the most common terms after removing stopwords
        terms = pd.Series(all_text_no_stopwords.split()).value_counts().head(3).index.tolist()
        
        # Create a description based on common terms
        cluster_name =f"{cluster}: " + " / ".join(terms) if terms else f"Cluster {cluster}"
        cluster_names[cluster] = cluster_name
        print(f"Cluster {cluster} name: {cluster_name}")

    # Map the descriptive names to each row in the DataFrame
    field_data['Cluster_Name'] = field_data['Cluster'].map(cluster_names)
    
    # Apply t-SNE to reduce to 2 dimensions for visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_components = tsne.fit_transform(embeddings_tensor)

    # Plot the clusters in 2D on the corresponding subplot
    for cluster in range(num_clusters):
        cluster_points = tsne_components[cluster_labels == cluster]
        axes[i].scatter(cluster_points[:, 0], cluster_points[:, 1], 
                        color=colors(cluster), label=f"{cluster_names[cluster]}")
    
    axes[i].set_title(f'{field}', fontsize=14)
    axes[i].set_xlabel('t-SNE Dimension 1')
    axes[i].set_ylabel('t-SNE Dimension 2')
    axes[i].legend(loc='upper right', bbox_to_anchor=(1.15, 1.05), fontsize='small')

    # Save the clustered data with names to a separate CSV for each field
    clustered_csv_path = f'clustered_patient_reports_{field}.csv'
    field_data.to_csv(clustered_csv_path, index=False)
    print(f"Clustering complete for {field}. Data saved to {clustered_csv_path}")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit title
plt.show()

# Print the cluster descriptions
print("\nCluster Descriptions by Field:")
for field in fields_to_plot:
    print(f"\nField: {field}")
    for cluster, name in cluster_names.items():
        print(f"  Cluster {cluster}: {name}")


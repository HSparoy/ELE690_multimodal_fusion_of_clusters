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


# Combine all fields into a single field
fields = ['Clinical_information', 'Desired_examination_treatment', 'Section_report', 'MRI_of_cor', 'Diagnosis']
patient_report_df['Combined_Text'] = patient_report_df[fields].fillna('').apply(
    lambda row: ' '.join(row.values), axis=1
)

# Remove stopwords from the combined text
patient_report_df['Combined_Text'] = patient_report_df['Combined_Text'].apply(remove_stopwords)

# Filter out rows with empty combined text
patient_report_df = patient_report_df[patient_report_df['Combined_Text'].str.strip() != '']

# Generate embeddings for each patient's combined information
embeddings = []
for text in patient_report_df['Combined_Text']:
    embedding = get_embeddings(text)
    embeddings.append(embedding)

# Stack all embeddings into a single tensor
embeddings_tensor = torch.cat(embeddings).numpy()

# Perform K-Means clustering 
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings_tensor)
patient_report_df['Cluster'] = cluster_labels

# Analyze each cluster to assign descriptive names
cluster_names = {}
for cluster in range(num_clusters):
    cluster_data = patient_report_df[patient_report_df['Cluster'] == cluster]
    combined_text = ' '.join(cluster_data['Combined_Text'])
    combined_text_no_stopwords = remove_stopwords(combined_text)
    
    # Find the most common terms in the cluster
    terms = pd.Series(combined_text_no_stopwords.split()).value_counts().head(5).index.tolist()
    cluster_name = f"{cluster}: " + " / ".join(terms) if terms else f"Cluster {cluster}"
    cluster_names[cluster] = cluster_name
    print(f"Cluster {cluster} name: {cluster_name}")

# Map cluster names to the DataFrame
patient_report_df['Cluster_Name'] = patient_report_df['Cluster'].map(cluster_names)

# Apply t-SNE to reduce to 2 dimensions for visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
tsne_components = tsne.fit_transform(embeddings_tensor)

# Plot the clusters in 2D
plt.figure(figsize=(10, 7))
colors = plt.cm.get_cmap('viridis', num_clusters)
for cluster in range(num_clusters):
    cluster_points = tsne_components[patient_report_df['Cluster'] == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                color=colors(cluster), label=f"{cluster_names[cluster]}")

plt.title('t-SNE Clusters Based on Combined Patient Text Information', fontsize=16)
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend(loc='upper right', fontsize='small', bbox_to_anchor=(1.3, 1))
plt.tight_layout()
plt.show()

# Save the clustered data to a CSV
clustered_csv_path = 'clustered_patient_reports_combined_tsne.csv'
patient_report_df.to_csv(clustered_csv_path, index=False)
print(f"Clustering complete. Data saved to {clustered_csv_path}")

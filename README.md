# ELE690_multimodal_fusion_of_clusters
Multimodal fusion of clusters based on MRI images of patients along with their patient journals

ran with Python version 3.10.11


Set the path in text_converter to where your documnet is. Run text to creata a CVS file.

To create a singel plot for the text clustering sett the cvs_path in ele690_p1_tsne.py to the path to  the CVS file created with text_converter. 
If you dont have 'stopword' dowloade uncomment 'nltk.download('stopwords')'  Run ele690_p1_tsne. 
If you want to change the default numer of cluster change to your wantet numer of cluster chang 'Cluster_num' to your wnated number of clusters
If you want to change the cluster title to include more or fewer word change the numer in 'head()' in terms = 'pd.Series(combined_text_no_stopwords.split()).value_counts().head(5).index.tolist() to your wanted number of key words.'

ele690_p5_tsne.py wokr the samme way as ele690_p1_tsne but has the added function of skiping filds. To skip a fild writ the name of the fild you want to skip in 'skip_fild'

To run ele690_association_matrix replace the path in text_clusters and image_clusters to the path to your clusters. If the colume names for patien id and clustrs are diffren in your CVS file, change them in the rename parth of the path.


patients.py needs to be ran to create the .csv file used for image clustering, paths in line 9 and 10 should be changed to match the location of the files.
when running pretrained_predictions.py line 69 in ELE690_functions.py needs to be uncommented, then commented again to work with trained_predictions.py.

to create clusters with any trained model run gmm.py and import the model in line 26. choose K-means by uncommenting lines 73-75 and commenting lines 67-69. Do the opposite for Gaussian mixture model.

In trained_predictions.py use K-Means for triplet loss training by uncommenting lines 99-102 and commenting lines 106-109 and make sure line 155 says pseudo_labels = kmeans.fit_predict(features).
To run Gaussian mixture model for triplet loss training uncomment lines 106-109 and comment lines 99-102 and make sure line 155 says pseudo_labels = gmm.fit_predict(features).
To use K-means for the final clustering uncomment lines 201-203 and comment lines 207-209. To use Gaussian mixture model for the final clustering uncomment lines 207-209 and comment lines 201-203.
To save the model from trained_predictions set the name in the variable model_name='x' on line 72. the model will then be named x.keras and the model after triplet loss tuning will be named x_tuned.keras. To spesifically save the cluster model used in the triplet loss training set the name on line 214.

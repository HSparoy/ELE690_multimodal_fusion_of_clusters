import pickle
import pandas as pd
import os

# create a dataframe of patients with their image paths.

patients = []
patients_path = []
base_path = '/home/prosjekt/BMDLab/data/mri/PM/' 
raw_path = '/home/prosjekt/BMDLab/data/mri/PM/erlend/p-files/'
for (dirpath, dirnames, filenames) in os.walk(raw_path):
    for i in filenames:
        path = raw_path + i
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        for key in [*obj.keys()]:
            patients.append(i[:-2])
            patients_path.append(base_path + key.replace('_', '/'))

df = pd.DataFrame({
    "patient": patients,
    "file": patients_path
})

df.to_csv('patients_and_paths.csv', index=False)

#with open('/home/prosjekt/BMDLab/data/mri/PM/erlend/p-files/PM009.p', 'rb') as f:
#   obj = pickle.load(f)
#patients = [fn.replace('_', '/') for fn in [*obj.keys()]]




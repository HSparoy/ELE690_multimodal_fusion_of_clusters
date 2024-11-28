#This code was providet by assosiate professor, not my originale work.
import docx
import pandas as pd
import re


''''''
def extract_features(text):
    features = {
        'Patient_Id': None,
        'Clinical_information': None,
        'Desired_examination_treatment': None,
        'Section_report': None,
        'MRI_of_cor': None,
        'Diagnosis': None
    }
    
   # Extract Patient ID
    patient_id_match = re.search(r'PM\d+', text)
    if patient_id_match:
        features['Patient_Id'] = patient_id_match.group()


 
    '''   patient_id_match = re.search(r'\bPM\d{3}\b', text)  # Matches 'PM' followed by exactly 3 digits
    if patient_id_match:
        features['Patient_Id'] = patient_id_match.group()
        print(f"Extracted Patient ID: {patient_id_match.group()}")  # Debugging output
    else:
        print("No Patient ID found in this paragraph.")  # Debugging output'''
    
    clinical_info_match = re.search(r'Kliniske opplysninger:\s*(.*?)(?=\n\s*\n|Ønsket|PM\d{3}|$)', text, re.DOTALL)
    if clinical_info_match:
        features['Clinical_information'] = clinical_info_match.group(1).strip()

    # Extract Desired_examination_treatment
    treatment_match = re.search(r'Ønsket undersøkelse/behandling:\s*(.*?)(?=\n\s*\n|Svarrapport|PM\d{3}|$)', text, re.DOTALL)
    if treatment_match:
        features['Desired_examination_treatment'] = treatment_match.group(1).strip()

    # Extract Section_report
    section_report_match = re.search(r'Svarrapport Seksjon\s*:\s*(.*?)(?=\n\s*\n|MR cor|MR av cor|MR hjerte|PM\d{3}|$)', text, re.DOTALL)
    if section_report_match:
        features['Section_report'] = section_report_match.group(1).strip()

    # Extract MRI of cor section
    mri_match = re.search(r'Svarrapport\s.*?(MR av cor|MR cor|MR hjerte)[:\s]+(.*?)(?:R:|$)', text, re.DOTALL | re.IGNORECASE)
    if mri_match:
        features['MRI_of_cor'] = mri_match.group(2).strip()

    # Extract Diagnosis (R:)
    diagnosis_match = re.search(r'^R:\s*(.*?)(?=\n\s*\n|\Z)', text, re.MULTILINE | re.DOTALL)    
    if diagnosis_match:
        features['Diagnosis'] = diagnosis_match.group(1).strip()
    
    return features




def read_patient_reports(file_path):
    # Load the document
    doc = docx.Document(file_path)
    all_features = []

    # Read each table in the document
    for table in doc.tables:
        for row in table.rows:
            # Each cell in the row contains the clinical report for a patient
            report = row.cells[0].text.strip()  # Assuming each patient report is in the first cell
            if report:  # Only process non-empty cells
                features = extract_features(report)
                all_features.append(features)

    # Create a DataFrame from all features
    df = pd.DataFrame(all_features)

    return df

# Extract Clinical Patient Report

#docx_file = '/nfs/student/astwol/PM/Tekst - Prosjekt MRI.docx'
docx_file = '/home/stud/astwol/Tekst - Prosjekt MRI.docx'
patient_report = read_patient_reports(docx_file)

# Display shape of the DataFrame
print(patient_report.shape)

# Example of selecting specific patient data
# Uncomment the lines below if the specified Patient IDs exist in the data
# print(patient_report.loc[patient_report['Patient_Id'] == 'PM269', ['MRI_of_cor', 'Diagnosis']].values[0])
# print(patient_report.loc[patient_report['Patient_Id'] == 'PM270', ['Diagnosis']].values[0])

# Display info and count of null values
patient_report.info()
print(patient_report.isnull().sum())

# Save the DataFrame to a CSV file
csv_path = 'clinical_notes_MRI_projcet.csv'
patient_report.to_csv(csv_path, index=False)
print(f"Data saved to {csv_path}")



# Print all Patient_Id values, including NaN
print("All Patient IDs in the dataset:")
print(patient_report['Patient_Id'])

# Print only non-null Patient IDs
print("\nNon-null Patient IDs in the dataset:")
print(patient_report['Patient_Id'].dropna())

# Print unique Patient IDs
print("\nUnique Patient IDs:")
print(patient_report['Patient_Id'].dropna().unique())

# Print counts of non-null and null Patient IDs
print("\nCount of non-null Patient IDs:", patient_report['Patient_Id'].notna().sum())
print("Count of null Patient IDs:", patient_report['Patient_Id'].isna().sum())

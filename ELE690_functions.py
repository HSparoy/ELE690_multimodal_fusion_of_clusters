import numpy as np
import pydicom
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.vgg16 import preprocess_input


def load_raw_image(path, reshape_size, dtype):
    with open(path, "rb") as f:
    
        raw_data = np.fromfile(f, dtype)

        desired_shape = reshape_size
        required_size = desired_shape[0] * desired_shape[1]

        # Crop the array if it's larger than the required size
        if raw_data.size > required_size:
            cropped_array = raw_data[:required_size]
            reshaped_image = cropped_array.reshape(desired_shape, order='F')


        #raw_data = raw_data.reshape(reshape_size, order="F").transpose()
        #raw_data = raw_data.reshape(reshape_size, order="F")

    
    return reshaped_image


def raw_img_processing(raw_string, tensor=False):
    if tensor:
        # Decode the filename tensor to a string
        raw_string = raw_string.numpy().decode('utf-8')

    # Load the raw image file 
    img = load_raw_image(raw_string, (224, 224), dtype=np.int16)

    # Normalize the image
    min_pixel = -32747
    max_pixel = 32736
    img = (img - min_pixel) / (max_pixel - min_pixel)

    return img

def dicom_img_processing(path, patient):

    path = path.numpy().decode('utf-8')
    dicom_data = pydicom.dcmread(path)
    image = dicom_data.pixel_array
    #image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32) / 255.0
    #image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    resized_image = cv2.resize(image, (224, 224))
    resized_image = np.expand_dims(resized_image, axis=-1)
    # should handle steps like normalization
    #image = preprocess_input(resized_image)
    return resized_image, patient


def tf_dicom_img_processing(path, patient_label):
    processed_image, patient_label = tf.py_function(
        func=dicom_img_processing,
        inp=[path, patient_label],
        Tout=(tf.float32, tf.int64) 
    )

    processed_image.set_shape((224, 224, 1))  # Image shape
    # mirror the channel to simulate rgb when pretrained model is in use
    #processed_image = tf.tile(processed_image, [1, 1, 3])
    patient_label.set_shape([])
    return processed_image, patient_label


def create_dataset(data, batch_size):
    
    dataset = tf.data.Dataset.from_tensor_slices((data['file'], data['patient_label'])) # might have to change to "patient" for pretrained model
    dataset = dataset.map(tf_dicom_img_processing, num_parallel_calls=tf.data.AUTOTUNE) 
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset


def create_patient_id_model(input_shape, num_patients):
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_patients, activation='softmax')  
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def triplet_loss(anchor, positive, negative, margin=1.0):
    # get the distances
    pos_distance = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_distance = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    loss = tf.maximum(0.0, pos_distance - neg_distance + margin)
    return tf.reduce_mean(loss)

def generate_triplets(features, labels):
    # returns every possible triplet
    triplets = []
    for i in range(len(features)):
        for j in range(len(features)):
            for k in range(len(features)):
                if labels[i] == labels[j] and labels[i] != labels[k]:  # Anchor-positive-negative condition
                    triplets.append((features[i], features[j], features[k]))
    return np.array(triplets)

def filter_hard_triplets(triplets):
    # return only the hard triplets from all inputtet triplets
    hard_triplets = []

    for triplet in triplets:
        anchor, positive, negative = triplet

        # Compute distances
        positive_distance = np.linalg.norm(anchor - positive)  # Anchor to Positive
        negative_distance = np.linalg.norm(anchor - negative)  # Anchor to Negative

        # Hard triplet condition: Positive is closer to Anchor than Negative
        if positive_distance < negative_distance:
            hard_triplets.append((anchor, positive, negative))
    
    return np.array(hard_triplets)

def generate_hard_triplets(features, labels):
    # generates all the hard triplets from the feature space
    triplets = []
    num_samples = len(features)

    for i in range(num_samples):
        anchor = features[i]
        anchor_label = labels[i]

        # Get positive and negative indices
        positive_indices = np.where(labels == anchor_label)[0]
        negative_indices = np.where(labels != anchor_label)[0]

        # Skip if not enough positives or negatives
        if len(positive_indices) < 2 or len(negative_indices) == 0:
            continue

        # Find the furthest positive form the anchor
        hardest_positive = None
        max_positive_dist = float('-inf')
        for pos_idx in positive_indices:
            if pos_idx != i:  # Skip the anchor itself
                positive = features[pos_idx]
                positive_dist = np.linalg.norm(anchor - positive)
                if positive_dist > max_positive_dist:
                    max_positive_dist = positive_dist
                    hardest_positive = positive

        # Find closest negative to anchor
        hardest_negative = None
        min_negative_dist = float('inf')
        for neg_idx in negative_indices:
            negative = features[neg_idx]
            negative_dist = np.linalg.norm(anchor - negative)
            if negative_dist < min_negative_dist:
                min_negative_dist = negative_dist
                hardest_negative = negative

        # Add the triplet
        if hardest_positive is not None and hardest_negative is not None:
            triplets.append((anchor, hardest_positive, hardest_negative))

    return np.array(triplets)

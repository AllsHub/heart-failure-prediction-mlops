"""
Module ini berfungsi untuk melakukan preprocessing data (Transform)
sebelum data masuk ke dalam proses training.
"""

# pylint: disable=import-error
import tensorflow as tf
import tensorflow_transform as tft

# --- Feature Definitions ---
CATEGORICAL_FEATURES = {
    'anaemia': 2,
    'diabetes': 2,
    'high_blood_pressure': 2,
    'sex': 2,
    'smoking': 2,
    'DEATH_EVENT': 2
}

NUMERICAL_FEATURES = [
    'age',
    'creatinine_phosphokinase',
    'ejection_fraction',
    'platelets',
    'serum_creatinine',
    'serum_sodium',
    'time'
]

LABEL_KEY = 'DEATH_EVENT'


# --- Naming Utility ---
def transformed_name(key):
    """Mengubah nama fitur asli menjadi nama fitur yang telah di-transform."""
    return key + '_xf'


# --- Preprocessing Function ---
def preprocessing_fn(inputs):
    """
    Fungsi callback utama untuk komponen Transform TFX.
    Melakukan scaling pada fitur numerik dan casting pada fitur kategorikal.

    Args:
        inputs: Dictionary berisi batch data input.

    Returns:
        outputs: Dictionary berisi data yang telah diproses.
    """
    outputs = {}

    for feature in NUMERICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_z_score(inputs[feature])

    for feature in CATEGORICAL_FEATURES:
        outputs[transformed_name(feature)] = tf.cast(inputs[feature], tf.int64)

    return outputs
    
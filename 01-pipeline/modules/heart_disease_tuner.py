# --- Imports ---
from typing import NamedTuple, Dict, Text, Any
from keras_tuner.engine import base_tuner
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs
import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft

# --- Feature Definitions ---
CATEGORICAL_FEATURES = {
    'anaemia': 2, 'diabetes': 2, 'high_blood_pressure': 2, 'sex': 2, 'smoking': 2
}
NUMERICAL_FEATURES = [
    'age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 
    'serum_creatinine', 'serum_sodium', 'time'
]
LABEL_KEY = 'DEATH_EVENT'

# --- Naming Utility ---
def transformed_name(key):
    return key + '_xf'

# --- Input Function ---
def input_fn(file_pattern, tf_transform_output, num_epochs=None, batch_size=32):
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())
    
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=lambda filenames: tf.data.TFRecordDataset(filenames, compression_type="GZIP"),
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY))
    
    return dataset

# --- Model Architecture ---
def model_builder(hp):
    inputs = {}
    for feature in NUMERICAL_FEATURES:
        inputs[transformed_name(feature)] = tf.keras.Input(
            shape=(1,), name=transformed_name(feature))
            
    for feature in CATEGORICAL_FEATURES:
        inputs[transformed_name(feature)] = tf.keras.Input(
            shape=(1,), name=transformed_name(feature))
            
    concatenated = layers.concatenate(list(inputs.values()))

    hp_units = hp.Int('units', min_value=32, max_value=256, step=32)
    x = layers.Dense(units=hp_units, activation='relu')(concatenated)
    
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    x = layers.Dropout(hp_dropout)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )
    
    return model

# --- Tuner Function ---
TunerFnResult = NamedTuple('TunerFnResult', [('tuner', base_tuner.BaseTuner),
                                             ('fit_kwargs', Dict[Text, Any])])

def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    
    train_set = input_fn(fn_args.train_files, tf_transform_output, num_epochs=10)
    eval_set = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=10)
    
    tuner = kt.RandomSearch(
        model_builder,
        objective='val_binary_accuracy',
        max_trials=20,
        directory=fn_args.working_dir,
        project_name='heart_disease_tuning',
        seed=42
    )
    
    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={ 
            'x': train_set,
            'validation_data': eval_set,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps
        }
    )
"""
Module ini berfungsi untuk melatih model Machine Learning (Trainer)
berdasarkan hyperparameter terbaik.
"""

# pylint: disable=import-error
# pylint: disable=duplicate-code
import os
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs

# --- Feature Definitions ---
CATEGORICAL_FEATURES = {
    'anaemia': 2, 'diabetes': 2, 'high_blood_pressure': 2, 'sex': 2, 'smoking': 2
}
NUMERICAL_FEATURES = [
    'age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets',
    'serum_creatinine', 'serum_sodium', 'time'
]
LABEL_KEY = 'DEATH_EVENT'


# --- Utilities ---
def transformed_name(key):
    """Mengubah nama fitur asli menjadi nama fitur yang telah di-transform."""
    return key + '_xf'


def gzip_reader_fn(filenames):
    """Fungsi utilitas untuk membaca file GZIP."""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


# --- Input Function ---
def input_fn(file_pattern, tf_transform_output, num_epochs=None, batch_size=32):
    """Fungsi untuk memuat dataset TFRecord."""
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY))

    return dataset


# --- Model Architecture ---
def model_builder(hyperparameters):
    """Membangun arsitektur model Keras untuk training."""
    inputs = {}
    for feature in NUMERICAL_FEATURES:
        inputs[transformed_name(feature)] = tf.keras.Input(
            shape=(1,), name=transformed_name(feature))

    for feature in CATEGORICAL_FEATURES:
        inputs[transformed_name(feature)] = tf.keras.Input(
            shape=(1,), name=transformed_name(feature))

    concatenated = layers.concatenate(list(inputs.values()))

    # Mengambil best hyperparameters atau default
    hp_units = hyperparameters.get('units') if hyperparameters else 32
    hp_dropout = hyperparameters.get('dropout') if hyperparameters else 0.1
    hp_lr = hyperparameters.get('learning_rate') if hyperparameters else 1e-3

    x_layer = layers.Dense(units=hp_units, activation='relu')(concatenated)
    x_layer = layers.Dropout(hp_dropout)(x_layer)
    x_layer = layers.Dense(64, activation='relu')(x_layer)
    x_layer = layers.Dense(32, activation='relu')(x_layer)

    outputs = layers.Dense(1, activation='sigmoid')(x_layer)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_lr),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    model.summary()
    return model


# --- Main Trainer Function ---
def run_fn(fn_args: FnArgs):
    """Fungsi utama untuk komponen Trainer."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = input_fn(fn_args.train_files, tf_transform_output, num_epochs=100)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=100)

    hyperparameters = fn_args.hyperparameters['values'] if fn_args.hyperparameters else None

    model = model_builder(hyperparameters)

    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq='batch'
    )

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback],
        epochs=10
    )

    signatures = {
        'serving_default':
        _get_serve_tf_examples_fn(model,
                                  tf_transform_output).get_concrete_function(
                                      tf.TensorSpec(
                                          shape=[None],
                                          dtype=tf.string,
                                          name='examples'))
    }

    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)


# --- Serving Function ---
def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Fungsi utilitas untuk serving model."""
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return serve_tf_examples_fn
    
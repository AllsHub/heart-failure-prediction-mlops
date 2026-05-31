"""
Script ini berfungsi untuk menjalankan TFX Pipeline menggunakan Apache Beam Orchestrator.
"""

import os
from typing import Text

from absl import logging
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

PIPELINE_NAME = 'aldomp7-pipeline'

# Pipeline Inputs
DATA_ROOT = 'data'
TRANSFORM_MODULE_FILE = 'modules/heart_disease_transform.py'
TUNER_MODULE_FILE = 'modules/heart_disease_tuner.py'
TRAINER_MODULE_FILE = 'modules/heart_disease_trainer.py'

# Pipeline Outputs
OUTPUT_BASE = 'output'
SERVING_MODEL_DIR = os.path.join(OUTPUT_BASE, 'serving_model')
PIPELINE_ROOT = os.path.join(OUTPUT_BASE, PIPELINE_NAME)
METADATA_PATH = os.path.join(PIPELINE_ROOT, 'metadata.sqlite')

def init_local_pipeline(
    components, pipeline_root: Text
) -> pipeline.Pipeline:
    """
    Fungsi untuk inisialisasi pipeline lokal dengan Apache Beam.
    """
    logging.info(f"Pipeline root set to: {pipeline_root}")
    
    # Argumen untuk Apache Beam
    beam_args = [
        "--direct_running_mode=in_memory",
        "--direct_num_workers=1"
    ]

    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            METADATA_PATH
        ),
        beam_pipeline_args=beam_args
    )

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)

    from modules.components import init_components

    # Inisialisasi komponen
    components = init_components(
        data_dir=DATA_ROOT,
        transform_module=TRANSFORM_MODULE_FILE,
        tuner_module=TUNER_MODULE_FILE,
        training_module=TRAINER_MODULE_FILE,
        training_steps=100,  # Bisa disesuaikan
        eval_steps=50,       # Bisa disesuaikan
        serving_model_dir=SERVING_MODEL_DIR,
    )

    # Inisialisasi dan jalankan pipeline
    pipeline = init_local_pipeline(components, PIPELINE_ROOT)
    BeamDagRunner().run(pipeline=pipeline)
    
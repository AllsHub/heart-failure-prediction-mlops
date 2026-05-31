"""
Module for initializing TFX pipeline components.
This module defines the init_components function which orchestrates
the creation of TFX components.
"""

# pylint: disable=import-error
# pylint: disable=too-many-arguments, too-many-locals, too-many-positional-arguments

import tensorflow_model_analysis as tfma
from tfx import v1 as tfx
from tfx.types import Channel
from tfx.dsl.components.common.resolver import Resolver
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy
)
from tfx.proto import pusher_pb2, trainer_pb2

def init_components(
    data_dir,
    transform_module,
    tuner_module,
    training_module,
    training_steps,
    eval_steps,
    serving_model_dir,
):
    """
    Initializes TFX pipeline components.

    Args:
        data_dir (str): Path to the input data directory.
        transform_module (str): Path to the transform module file.
        tuner_module (str): Path to the tuner module file.
        training_module (str): Path to the trainer module file.
        training_steps (int): Number of training steps.
        eval_steps (int): Number of evaluation steps.
        serving_model_dir (str): Path to the serving model directory.

    Returns:
        List[tfx.components.BaseComponent]: A list of initialized TFX components.
    """

    # 1. ExampleGen
    example_gen = tfx.components.CsvExampleGen(input_base=data_dir)

    # 2. StatisticsGen
    statistics_gen = tfx.components.StatisticsGen(
        examples=example_gen.outputs['examples']
    )

    # 3. SchemaGen
    schema_gen = tfx.components.SchemaGen(
        statistics=statistics_gen.outputs['statistics']
    )

    # 4. ExampleValidator
    example_validator = tfx.components.ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )

    # 5. Transform
    transform = tfx.components.Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=transform_module
    )

    # 6. Tuner
    tuner = tfx.components.Tuner(
        module_file=tuner_module,
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(num_steps=training_steps),
        eval_args=trainer_pb2.EvalArgs(num_steps=eval_steps)
    )

    # 7. Trainer
    trainer = tfx.components.Trainer(
        module_file=training_module,
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        hyperparameters=tuner.outputs['best_hyperparameters'],
        train_args=trainer_pb2.TrainArgs(num_steps=training_steps),
        eval_args=trainer_pb2.EvalArgs(num_steps=eval_steps)
    )

    # 8. Resolver
    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    ).with_id('latest_blessed_model_resolver')

    # 9. Evaluator
    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key='DEATH_EVENT')],
        slicing_specs=[
            tfma.SlicingSpec(),
            tfma.SlicingSpec(feature_keys=['sex'])
        ],
        metrics_specs=[
            tfma.MetricsSpec(metrics=[
                tfma.MetricConfig(class_name='ExampleCount'),
                tfma.MetricConfig(
                    class_name='BinaryAccuracy',
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={'value': 0.5}
                        ),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value': 1e-10}
                        )
                    )
                )
            ])
        ]
    )

    evaluator = tfx.components.Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config
    )

    # 10. Pusher
    pusher = tfx.components.Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        )
    )

    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        tuner,
        trainer,
        model_resolver,
        evaluator,
        pusher
    ]

    return components

import kfp
from kfp import dsl
import yaml
import json
import kfp.components as comp
from kfp.v2 import compiler 

import google.cloud.aiplatform as aip

PIPELINE_ROOT="gs://leafy-thunder-389112-machine-learning-solene/pipeline_root/intro"
REGION='europe-west1'
DISPLAY_NAME = "intro-pipeline-job-unique"

TRAIN=False
MODEL_DIR_IF_NO_TRAIN="level-1-models/temp/processor_state.pkl"

@dsl.pipeline(
    name='finalpipeline',
    description='An example pipeline.'
)
def ML_Pipeline(train:bool=TRAIN):
    """
    Define the Kubeflow pipeline with the Processing step
    """

    preprocess_yaml = 'preprocess.yaml'
    preprocess_config = 'preprocess_config.json'
    train_yaml = 'train.yaml'
    train_config = 'train_config.json'
    test_yaml = 'test.yaml'
    test_config = 'test_config.json'

    _preprocess_op = comp.load_component_from_file(preprocess_yaml)
    with open(preprocess_config, 'r') as f:
        preprocess_params = json.load(f)
        print(preprocess_params)
    preprocess = _preprocess_op(**preprocess_params)

    with dsl.Condition(train == True, name='is-true'):
        _train_op = comp.load_component_from_file(train_yaml)
        with open(train_config, 'r') as f:
            train_params = json.load(f)
            print(train_params)
        train = _train_op(preprocess_data_dir=preprocess.output, **train_params)

        _test_op = comp.load_component_from_file(test_yaml)
        with open(test_config, 'r') as f:
            test_params = json.load(f)
            print(test_params)
        test = _test_op(preprocess_data_dir=preprocess.output, model_dir=train.output, **test_params)

    with dsl.Condition(train == False, name='is-false'):
        _test_op = comp.load_component_from_file(test_yaml)
        with open(test_config, 'r') as f:
            test_params = json.load(f)
            print(test_params)
        test = _test_op(preprocess_data_dir=preprocess.output, model_dir=MODEL_DIR_IF_NO_TRAIN, **test_params)


    # _preprocess_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    # _test_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    # _train_op.execution_options.caching_strategy.max_cache_staleness = "P0D"


compiler.Compiler().compile(pipeline_func=ML_Pipeline, package_path="intro_pipeline.json")

job = aip.PipelineJob(
    display_name=DISPLAY_NAME,
    template_path="intro_pipeline.json",
    pipeline_root=PIPELINE_ROOT,
    location=REGION
)

job.run()





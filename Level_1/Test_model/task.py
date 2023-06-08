import argparse
import json
import test_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Vertex custom container training args. These are set by Vertex AI during training but can also be overwritten.
    parser.add_argument('--model-dir', dest='model-dir',
                        type=str, help='Model dir.')
    parser.add_argument('--preprocess-data-dir', dest='preprocess-data-dir', type=str,
                        help='Training data GCS or BQ URI set during Vertex AI training.')
    #
    parser.add_argument('--model-validation-dir', dest='model-validation-dir',
                        type=str, help='valid Model dir.')
    #
    parser.add_argument('--performance-threshold', dest='performance-threshold', default='{"acc":0.75, "std":0.3}', type=str, help='valid Model dir.')
    parser.add_argument('--output-performance-model', dest='output-performance-model',
                        type=str, help='valid Model dir.')


    args = parser.parse_args()

    hparams = args.__dict__

    test_model.evaluate_model(hparams)

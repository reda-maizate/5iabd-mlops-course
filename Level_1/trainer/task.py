import argparse

import trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Vertex custom container training args. These are set by Vertex AI during training but can also be overwritten.
    parser.add_argument('--model-dir', dest='model-dir',
                        type=str, help='Model dir.')
    parser.add_argument('--bucket', dest='bucket', type=str, help='bucket name.')
    parser.add_argument('--preprocess-data-dir', dest='preprocess-data-dir', type=str,
                        help='Training data GCS or BQ URI set during Vertex AI training.')
    parser.add_argument('--temp-dir', dest='temp-dir', type=str, help='Temp dir set during Vertex AI training.')
    # Model training args.
    parser.add_argument('--batch-size', dest='batch-size', default=128, type=int,
                        help='batch size.')
    parser.add_argument('--filters', dest='filters', default=64, type=int,
                        help='nb filters')
    parser.add_argument('--dropout', dest='dropout', default=0.5, type=float,
                        help='dropout')
    parser.add_argument('--n-checkpoints', dest='n-checkpoints', default=1, type=int,
                        help='nb of checkpoints')

    parser.add_argument('--output-model-dir', dest='output-model-dir', type=str,
                        help='Output file')

    args = parser.parse_args()

    hparams = args.__dict__

    trainer.train_evaluate_explain_model(hparams)
"""A streaming python pipeline to read in pubsub tweets and perform classification"""
from __future__ import absolute_import

import argparse
import json
import logging

import apache_beam as beam
import apache_beam.transforms.window as window
import yaml
from apache_beam.io.gcp.bigquery_tools import parse_table_schema_from_json
from apache_beam.options.pipeline_options import (GoogleCloudOptions,
                                                  PipelineOptions,
                                                  SetupOptions,
                                                  StandardOptions)
from apache_beam.transforms.util import BatchElements
from googleapiclient import discovery

cmle_api = None

with open("config.yml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)
    PROJECT_ID = cfg['project-id']
    STAGING_LOCATION = cfg['staging-location']
    TEMP_LOCATION = cfg['temp-location']
    REGION = cfg['region']
    TABLE = cfg['table']
    DATASET = cfg['dataset']
    SUBSCRIPTION = cfg['subscription']
    MODEL_NAME = cfg['model_name']


def init_api():
    global cmle_api
    # If it hasn't been instantiated yet: do it now
    if cmle_api is None:
        cmle_api = discovery.build('ml', 'v1')


def estimate_cmle(instances):
    """
    Calls the tweet_sentiment_classifier API on CMLE to get predictions
    Args:
       instances: list of strings
    Returns:
        float: estimated values
    """
    init_api()
    request_data = {'instances': instances}
    logging.info("making request to the ML api")
    # Call the model
    model_url = 'projects/{}/models/{}'.format(PROJECT_ID, MODEL_NAME)
    response = cmle_api.projects().predict(body=request_data, name=model_url).execute()
    # Read out the scores
    values = [item["score"] for item in response['predictions']]

    return values


def estimate(messages):
    # Be able to cope with a single string as well
    if not isinstance(messages, list):
        messages = [messages]

    # Messages from pubsub are JSON strings
    instances = list(map(lambda message: json.loads(message), messages))

    # Estimate the sentiment of the 'text' of each tweet
    scores = estimate_cmle([instance["text"] for instance in instances])

    # Join them together
    for i, instance in enumerate(instances):
        instance['sentiment'] = scores[i]

    logging.info("first message in batch")
    logging.info(instances[0])

    return instances


class MyOptions(PipelineOptions):
    @classmethod
    def _add_argparse_args(cls, parser):
        parser.add_argument(
            '--input_subscription',
            help=('Input PubSub subscription of the form '
                  '"projects/<PROJECT>/subscriptions/<SUBSCRIPTION>."'),
            default="projects/{}/subscriptions/{}".format(PROJECT_ID, SUBSCRIPTION)
        )


def run(argv=None):
    parser = argparse.ArgumentParser()
    known_args, pipeline_args = parser.parse_known_args(argv)

    # Create the options object
    options = PipelineOptions(pipeline_args)
    tweet_options = options.view_as(MyOptions)

    google_cloud_options = options.view_as(GoogleCloudOptions)
    google_cloud_options.project = PROJECT_ID
    google_cloud_options.staging_location = STAGING_LOCATION
    google_cloud_options.temp_location = TEMP_LOCATION
    google_cloud_options.region = REGION

    options.view_as(SetupOptions).save_main_session = True
    options.view_as(StandardOptions).streaming = True
    options.view_as(StandardOptions).runner = 'DataflowRunner'

    bigqueryschema_json = '{"fields": [' \
                          '{"name":"id","type":"STRING"},' \
                          '{"name":"text","type":"STRING"},' \
                          '{"name":"user_id","type":"STRING"},' \
                          '{"name":"sentiment","type":"FLOAT"},' \
                          '{"name":"created_at","type":"STRING"}' \
                          ']}'

    bigqueryschema = parse_table_schema_from_json(bigqueryschema_json)

    p = beam.Pipeline(options=options)

    (p
     | "read in tweets" >> beam.io.ReadFromPubSub(
                subscription=tweet_options.input_subscription,
                with_attributes=False,
                id_label="id")
     | 'assign window key' >> beam.WindowInto(window.FixedWindows(10))
     | 'batch into n batches' >> BatchElements(min_batch_size=49, max_batch_size=50)
     | 'predict sentiment' >> beam.FlatMap(lambda messages: estimate(messages))
     | 'store twitter posts' >> beam.io.WriteToBigQuery(
                table=TABLE,  # change
                dataset=DATASET,
                schema=bigqueryschema,
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                project=PROJECT_ID)
     )

    p.run()


if __name__ == '__main__':
    run()

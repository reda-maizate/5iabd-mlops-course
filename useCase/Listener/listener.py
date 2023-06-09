"""Retrieve live chat content from YouTube videos."""

import pytchat
from google.cloud import pubsub_v1
import json
import yaml

# Method to push messages to pub/sub
def write_to_pubsub(data):
    try:
        publisher.publish(topic_path, data=json.dumps({
            "text": data.message,
            "user_id": data.author.name,
            "id": data.id,
            "created_at": data.datetime
        }).encode("utf-8"))
    except Exception as e:
        raise

with open("config.yml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)
    PROJECT_ID = cfg['project_id']
    TOPIC_NAME = cfg['topic_name']
    YOUTUBE_URL = cfg['youtube_url']

# Pub/Sub topic configuration
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(PROJECT_ID, TOPIC_NAME)

chat = pytchat.create(video_id=YOUTUBE_URL)
while chat.is_alive() :
    for data in chat.get().sync_items():
        print(f"{data.datetime} [{data.author.name}]- {data.message}")
        write_to_pubsub(data)


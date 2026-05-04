import pika
import json
import uuid

connection = pika.BlockingConnection(
    pika.ConnectionParameters(
        host='192.168.200.165',
        port=18011,
        credentials=pika.PlainCredentials(
            username='guest',
            password='guest',
        )
    ),
)
channel = connection.channel()

message = {
    "uuid": str(uuid.uuid4()),
    "file_path": "/home/pouya/Desktop/axonnegar_ai_modes.png",
    "model_name": "gpt-4o",
    "lang": "fa"
}

channel.basic_publish(
    exchange='',
    routing_key='ocr_input_queue',
    body=json.dumps(message)
)

print(" [x] Sent 1 job to ocr_input_queue")
connection.close()
import pika
import json

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

def callback(ch, method, properties, body):
    print(" [x] Received Result from Output Queue:")
    print(json.dumps(json.loads(body.decode()), indent=4, ensure_ascii=False))

channel.basic_consume(queue='ocr_output_queue', on_message_callback=callback, auto_ack=True)

print(' [*] Waiting for OCR result...')
channel.start_consuming()
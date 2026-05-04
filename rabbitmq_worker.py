import json, uuid, pika, threading, uvicorn, asyncio, os
from fastapi import FastAPI
from contextlib import asynccontextmanager
from logger_config import logger
from ocr_manager import process_image_to_text
from dotenv import load_dotenv


load_dotenv()


def process_message(ch, method, properties, body):
    print(" [!] I received a message!")
    data = json.loads(body.decode())
    req_id = data.get("uuid", str(uuid.uuid4()))
    try:
        text = asyncio.run(process_image_to_text(data["file_path"], data["model_name"], data.get("lang", "en")))
        result = {"id": req_id, "status": "success", "text": text}
    except Exception as e:
        logger.error(f"Error: {e}")
        result = {"id": req_id, "status": "failed", "error": str(e)}

    ch.basic_publish(exchange="", routing_key="ocr_output_queue", body=json.dumps(result))
    ch.basic_ack(delivery_tag=method.delivery_tag)


def run_worker():
    host = os.getenv("RABBITMQ_HOST", "192.168.200.165")
    port = int(os.getenv("RABBITMQ_PORT", "18011"))

    conn = pika.BlockingConnection(pika.ConnectionParameters(host=host, port=port))
    ch = conn.channel()
    ch.queue_declare(queue="ocr_input_queue")
    ch.queue_declare(queue="ocr_output_queue")
    ch.basic_consume(queue="ocr_input_queue", on_message_callback=process_message)
    logger.info(f"Worker connected to {host}:{port}")
    ch.start_consuming()


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=run_worker, daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
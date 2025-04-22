from confluent_kafka import Producer
from fast_template import build_random_message
import time

producer = Producer({'bootstrap.servers': 'localhost:9092'})

def delivery_report(err, msg):
    if err:
        print(f"[❌] Delivery failed: {err}")
    else:
        print(f"[✅] Sent message to {msg.topic()} [{msg.partition()}]")

def main():
    topic = "fast_topic"
    print(f"[🚀] Producing FAST messages to topic: {topic}")
    while True:
        msg = build_random_message()
        producer.produce(topic, msg, callback=delivery_report)
        producer.poll(0)
        time.sleep(0.01)

if __name__ == "__main__":
    main()

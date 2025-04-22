from confluent_kafka import Producer
from itch_template import build_random_itch
import time

producer = Producer({'bootstrap.servers': 'localhost:9092'})

def main():
    topic = "itch_topic"
    print("[🚀] Producing ITCH messages...")
    while True:
        msg = build_random_itch()
        producer.produce(topic, msg)
        producer.poll(0)
        time.sleep(0.01)

if __name__ == "__main__":
    main()

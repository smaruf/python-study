from confluent_kafka import Producer
from fix_template import build_fix_message
import time

producer = Producer({'bootstrap.servers': 'localhost:9092'})

def main():
    topic = "fix_topic"
    print("[🚀] Producing FIX messages...")
    while True:
        msg = build_fix_message()
        producer.produce(topic, msg)
        producer.poll(0)
        time.sleep(0.01)

if __name__ == "__main__":
    main()

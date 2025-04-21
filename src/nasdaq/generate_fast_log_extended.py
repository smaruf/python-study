import struct
import random
import time

# Constants for message types
MESSAGE_TYPES = {
    "ORDER_ADD": 1,
    "ORDER_EXEC": 2,
    "ORDER_CANCEL": 3,
    "DEPTH_UPDATE": 4,
    "MARKET_STATUS": 5,
    "SPECIAL_NEWS": 6
}

def random_string(length=20):
    return ''.join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ", k=length))

def encode_message(msg_type, data):
    """
    Encodes different message types into binary
    Structure:
    - Message Type (1 byte)
    - Timestamp (8 bytes, float)
    Then conditional fields based on msg_type
    """
    ts = time.time()
    base = struct.pack('Bd', msg_type, ts)

    if msg_type == MESSAGE_TYPES["ORDER_ADD"]:
        return base + struct.pack('IIf', data['order_id'], data['volume'], data['price'])
    elif msg_type == MESSAGE_TYPES["ORDER_EXEC"]:
        return base + struct.pack('II', data['order_id'], data['exec_qty'])
    elif msg_type == MESSAGE_TYPES["ORDER_CANCEL"]:
        return base + struct.pack('I', data['order_id'])
    elif msg_type == MESSAGE_TYPES["DEPTH_UPDATE"]:
        return base + struct.pack('Iff', data['depth_level'], data['bid'], data['ask'])
    elif msg_type == MESSAGE_TYPES["MARKET_STATUS"]:
        return base + struct.pack('B', data['status'])  # 0=closed, 1=open, 2=halt
    elif msg_type == MESSAGE_TYPES["SPECIAL_NEWS"]:
        news = data['headline'].encode("utf-8")[:50]
        return base + struct.pack(f'{len(news)}s', news.ljust(50, b'\x00'))
    else:
        return base  # fallback

def generate_random_message():
    msg_type = random.choice(list(MESSAGE_TYPES.values()))
    if msg_type == MESSAGE_TYPES["ORDER_ADD"]:
        return msg_type, {
            "order_id": random.randint(1000, 99999),
            "volume": random.randint(10, 1000),
            "price": round(random.uniform(50, 200), 2)
        }
    elif msg_type == MESSAGE_TYPES["ORDER_EXEC"]:
        return msg_type, {
            "order_id": random.randint(1000, 99999),
            "exec_qty": random.randint(1, 500)
        }
    elif msg_type == MESSAGE_TYPES["ORDER_CANCEL"]:
        return msg_type, {
            "order_id": random.randint(1000, 99999)
        }
    elif msg_type == MESSAGE_TYPES["DEPTH_UPDATE"]:
        return msg_type, {
            "depth_level": random.randint(1, 5),
            "bid": round(random.uniform(50, 150), 2),
            "ask": round(random.uniform(150, 200), 2)
        }
    elif msg_type == MESSAGE_TYPES["MARKET_STATUS"]:
        return msg_type, {
            "status": random.choice([0, 1, 2])  # 0=closed, 1=open, 2=halt
        }
    elif msg_type == MESSAGE_TYPES["SPECIAL_NEWS"]:
        return msg_type, {
            "headline": random_string()
        }

def create_fast_log(filename="fast_extended_10k.log", count=10000):
    with open(filename, "wb") as f:
        for _ in range(count):
            msg_type, data = generate_random_message()
            encoded = encode_message(msg_type, data)
            f.write(encoded)
    print(f"[✓] FAST-style log created with {count} messages: {filename}")

if __name__ == "__main__":
    create_fast_log()

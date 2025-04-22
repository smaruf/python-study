import struct
import random
import time

TEMPLATE = {
    "OrderAdd": [("order_id", "I"), ("volume", "I"), ("price", "f")],
    "OrderExec": [("order_id", "I"), ("exec_qty", "I")],
    "OrderCancel": [("order_id", "I")],
    "DepthUpdate": [("depth_level", "I"), ("bid", "f"), ("ask", "f")],
    "MarketStatus": [("status", "B")],
    "SpecialNews": [("headline", "50s")],
    "MDByPrice": [("symbol", "10s"), ("entry_type", "B"), ("price", "f"), ("size", "I"), ("position", "I")]
}

MESSAGE_TYPES = {
    "OrderAdd": 1,
    "OrderExec": 2,
    "OrderCancel": 3,
    "DepthUpdate": 4,
    "MarketStatus": 5,
    "SpecialNews": 6,
    "MDByPrice": 7
}

def build_message(msg_type: str):
    timestamp = time.time()
    base = struct.pack("Bd", MESSAGE_TYPES[msg_type], timestamp)
    fields = TEMPLATE[msg_type]
    values = []

    for field, fmt in fields:
        if fmt == "I":
            if "size" in field or "exec_qty" in field or "volume" in field:
                values.append(random.randint(1, 1000))
            elif "order_id" in field:
                values.append(random.randint(10000, 99999))
            elif "position" in field:
                values.append(random.randint(1, 5))
        elif fmt == "f":
            values.append(random.uniform(50, 200))
        elif fmt == "B":
            if "entry_type" in field:
                values.append(random.randint(0, 1))  # 0 = Bid, 1 = Ask
            else:
                values.append(random.randint(0, 2))  # status
        elif fmt == "10s":
            symbol = random.choice(["ACI", "BXPHARMA", "SQURPHARMA"]).encode("utf-8").ljust(10, b'\x00')
            values.append(symbol)
        elif fmt == "50s":
            headline = random.choice(["Breaking News", "Market Halt", "Earnings Call"]).ljust(50, '\x00').encode("utf-8")
            values.append(headline)

    body = struct.pack("".join(fmt for _, fmt in fields), *values)
    return base + body

def generate_template_based_log(filename="fast_template_encoded.log", count=10000):
    with open(filename, "wb") as f:
        for _ in range(count):
            msg_type = random.choice(list(TEMPLATE.keys()))
            msg = build_message(msg_type)
            f.write(msg)
    print(f"[✓] FAST template-style log written to {filename}")

if __name__ == "__main__":
    generate_template_based_log()

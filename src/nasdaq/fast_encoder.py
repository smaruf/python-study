import struct
import random
import time

output_file = "fast_extended_10k.log"

MESSAGE_TYPES = {
    "ORDER_ADD": 1,
    "ORDER_EXEC": 2,
    "ORDER_CANCEL": 3,
    "DEPTH_UPDATE": 4,
    "MARKET_STATUS": 5,
    "SPECIAL_NEWS": 6,
    "MD_BY_PRICE": 7
}

def generate_message():
    msg_type = random.choice(list(MESSAGE_TYPES.values()))
    timestamp = time.time()
    base = struct.pack("Bd", msg_type, timestamp)

    if msg_type == 1:
        order_id = random.randint(10000, 99999)
        volume = random.randint(10, 1000)
        price = round(random.uniform(50, 200), 2)
        return base + struct.pack("IIf", order_id, volume, price)

    elif msg_type == 2:
        order_id = random.randint(10000, 99999)
        exec_qty = random.randint(1, 500)
        return base + struct.pack("II", order_id, exec_qty)

    elif msg_type == 3:
        order_id = random.randint(10000, 99999)
        return base + struct.pack("I", order_id)

    elif msg_type == 4:
        level = random.randint(1, 5)
        bid = round(random.uniform(90, 150), 2)
        ask = bid + round(random.uniform(0.1, 1.0), 2)
        return base + struct.pack("Iff", level, bid, ask)

    elif msg_type == 5:
        status = random.randint(0, 2)  # 0=CLOSED, 1=OPEN, 2=HALT
        return base + struct.pack("B", status)

    elif msg_type == 6:
        headlines = ["MARKET OPENS STRONG", "TRADING HALTED", "NEWS: RATE CUT", "DIVIDEND DECLARED"]
        headline = random.choice(headlines).ljust(50, '\x00').encode("utf-8")
        return base + headline

    elif msg_type == 7:
        symbol = random.choice(["SQURPHARMA", "BXPHARMA", "ACI"])
        entry_type = random.choice([0, 1])  # 0 = Bid, 1 = Ask
        price = round(random.uniform(100, 200), 2)
        size = random.randint(100, 10000)
        position = random.randint(1, 5)
        symbol_bytes = symbol.encode("utf-8")[:10].ljust(10, b'\x00')
        return base + symbol_bytes + struct.pack('BfII', entry_type, price, size, position)

with open(output_file, "wb") as f:
    for _ in range(10000):
        f.write(generate_message())

print(f"[✓] FAST data log generated with 10,000 entries → {output_file}")

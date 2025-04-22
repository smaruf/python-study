import struct
import random
import time

TEMPLATE = {
    1: ("OrderAdd", [("order_id", "I"), ("volume", "I"), ("price", "f")]),
    2: ("OrderExec", [("order_id", "I"), ("exec_qty", "I")]),
    3: ("OrderCancel", [("order_id", "I")]),
    4: ("DepthUpdate", [("depth_level", "I"), ("bid", "f"), ("ask", "f")]),
    5: ("MarketStatus", [("status", "B")]),
    6: ("SpecialNews", [("headline", "50s")]),
    7: ("MDByPrice", [("symbol", "10s"), ("entry_type", "B"), ("price", "f"), ("size", "I"), ("position", "I")])
}

def build_random_message():
    msg_type = random.randint(1, 7)
    timestamp = time.time()
    msg_name, fields = TEMPLATE[msg_type]
    values = []

    for field, fmt in fields:
        if fmt == "I":
            values.append(random.randint(1, 9999))
        elif fmt == "f":
            values.append(round(random.uniform(50.0, 200.0), 2))
        elif fmt == "B":
            values.append(random.randint(0, 2))
        elif fmt == "10s":
            values.append(random.choice(["ACI", "BXPHARMA", "SQURPHARMA"]).ljust(10, "\x00").encode())
        elif fmt == "50s":
            values.append(random.choice(["Market Halt", "Breaking News", "Earnings Call"]).ljust(50, "\x00").encode())

    header = struct.pack("Bd", msg_type, timestamp)
    body = struct.pack("".join(fmt for _, fmt in fields), *values)
    return header + body

def decode_message(binary):
    if len(binary) < 9:
        return None
    msg_type, ts = struct.unpack("Bd", binary[:9])
    msg_name, fields = TEMPLATE.get(msg_type, ("Unknown", []))
    fmt = "".join(fmt for _, fmt in fields)
    body = binary[9:]
    values = struct.unpack(fmt, body)
    data = {"type": msg_name, "timestamp": ts}
    for (key, fmt), val in zip(fields, values):
        if isinstance(val, bytes):
            val = val.decode("utf-8").strip("\x00")
        elif key == "entry_type":
            val = "Bid" if val == 0 else "Ask"
        elif key == "status":
            val = ["CLOSED", "OPEN", "HALT"][val] if val < 3 else "UNKNOWN"
        data[key] = val
    return data

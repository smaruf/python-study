import struct
import random
import time

# Message types
TEMPLATE = {
    b'A': ("AddOrder", [('order_id', 'Q'), ('side', 'c'), ('shares', 'I'), ('stock', '8s'), ('price', 'I')]),
    b'E': ("OrderExecuted", [('order_id', 'Q'), ('executed_shares', 'I')]),
    b'X': ("OrderCancelled", [('order_id', 'Q'), ('cancelled_shares', 'I')]),
    b'P': ("Trade", [('trade_id', 'Q'), ('stock', '8s'), ('price', 'I'), ('shares', 'I')]),
    b'S': ("SystemEvent", [('event_code', 'c')])
}

def build_random_itch():
    msg_type = random.choice(list(TEMPLATE.keys()))
    name, fields = TEMPLATE[msg_type]
    values = []

    for field, fmt in fields:
        if fmt == 'Q':
            values.append(random.randint(1000000, 9999999))
        elif fmt == 'I':
            values.append(random.randint(1, 1000))
        elif fmt == 'c':
            if field == 'side':
                values.append(random.choice([b'B', b'S']))
            elif field == 'event_code':
                values.append(random.choice([b'O', b'C', b'H']))
        elif fmt == '8s':
            values.append(random.choice(['AAPL', 'MSFT', 'TSLA']).ljust(8).encode())

    header = msg_type
    body = struct.pack("".join(fmt for _, fmt in fields), *values)
    return header + body

def decode_itch(data):
    if not data:
        return None
    msg_type = data[0:1]
    name, fields = TEMPLATE.get(msg_type, ("Unknown", []))
    fmt = "".join(fmt for _, fmt in fields)
    try:
        values = struct.unpack(fmt, data[1:])
        decoded = {'type': name}
        for (field, fmt), val in zip(fields, values):
            if isinstance(val, bytes):
                val = val.decode().strip()
            decoded[field] = val
        return decoded
    except Exception as e:
        return {"error": str(e)}

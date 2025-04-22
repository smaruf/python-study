import random
import time

# Basic sample tags (simplified FIX 4.2 subset)
TAGS = {
    35: ["D", "F", "8", "X"],  # MsgType
    55: ["AAPL", "GOOG", "MSFT", "TSLA"],  # Symbol
    54: ["1", "2"],  # Side (1 = Buy, 2 = Sell)
    38: lambda: str(random.randint(10, 1000)),  # OrderQty
    44: lambda: f"{random.uniform(100, 300):.2f}",  # Price
    60: lambda: time.strftime("%Y%m%d-%H:%M:%S"),  # TransactTime
}

SOH = '\x01'

def build_fix_message():
    msg_type = random.choice(TAGS[35])
    symbol = random.choice(TAGS[55])
    side = random.choice(TAGS[54])
    qty = TAGS[38]()
    price = TAGS[44]()
    transact_time = TAGS[60]()

    body = [
        f"8=FIX.4.2",
        f"35={msg_type}",
        f"55={symbol}",
        f"54={side}",
        f"38={qty}",
        f"44={price}",
        f"60={transact_time}"
    ]
    body_str = SOH.join(body) + SOH
    body_len = len(body_str)
    msg = f"8=FIX.4.2{SOH}9={body_len}{SOH}{body_str}10=000{SOH}"
    return msg.encode()

def parse_fix_message(raw_msg):
    try:
        parts = raw_msg.decode().strip().split(SOH)
        kv = dict(item.split('=') for item in parts if '=' in item)
        return kv
    except Exception as e:
        return {"error": str(e)}

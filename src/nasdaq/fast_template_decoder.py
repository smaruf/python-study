import struct
import json
from datetime import datetime

TEMPLATE = {
    1: ("OrderAdd", [("order_id", "I"), ("volume", "I"), ("price", "f")]),
    2: ("OrderExec", [("order_id", "I"), ("exec_qty", "I")]),
    3: ("OrderCancel", [("order_id", "I")]),
    4: ("DepthUpdate", [("depth_level", "I"), ("bid", "f"), ("ask", "f")]),
    5: ("MarketStatus", [("status", "B")]),
    6: ("SpecialNews", [("headline", "50s")]),
    7: ("MDByPrice", [("symbol", "10s"), ("entry_type", "B"), ("price", "f"), ("size", "I"), ("position", "I")])
}

def decode_template_based_log(filename="fast_template_encoded.log", output_json="decoded_template.json"):
    messages = []

    with open(filename, "rb") as f:
        while True:
            header = f.read(9)
            if not header or len(header) < 9:
                break
            msg_type, ts = struct.unpack("Bd", header)
            timestamp = datetime.fromtimestamp(ts).isoformat()

            if msg_type not in TEMPLATE:
                continue

            msg_name, fields = TEMPLATE[msg_type]
            fmt = "".join(f[1] for f in fields)
            body_size = struct.calcsize(fmt)
            body = f.read(body_size)
            if len(body) < body_size: break

            values = struct.unpack(fmt, body)
            decoded = {"timestamp": timestamp, "type": msg_name}
            for (field_name, _), value in zip(fields, values):
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="ignore").strip("\x00")
                if field_name == "entry_type":
                    value = "Bid" if value == 0 else "Ask"
                elif field_name == "status":
                    value = ["CLOSED", "OPEN", "HALT"][value] if value < 3 else "UNKNOWN"
                decoded[field_name] = value

            messages.append(decoded)

    with open(output_json, "w") as f:
        json.dump(messages, f, indent=2)

    print(f"[✓] Decoded JSON written to {output_json}")

if __name__ == "__main__":
    decode_template_based_log()

import struct
import json
from datetime import datetime

MESSAGE_TYPES = {
    1: "ORDER_ADD",
    2: "ORDER_EXEC",
    3: "ORDER_CANCEL",
    4: "DEPTH_UPDATE",
    5: "MARKET_STATUS",
    6: "SPECIAL_NEWS",
    7: "MD_BY_PRICE"
}

def decode_log_to_json(input_file="fast_extended_10k.log", output_file="decoded_fast_log.json"):
    messages = []

    with open(input_file, "rb") as f:
        while True:
            header = f.read(9)
            if not header or len(header) < 9:
                break

            msg_type, ts = struct.unpack("Bd", header)
            timestamp = datetime.fromtimestamp(ts).isoformat()
            msg_type_str = MESSAGE_TYPES.get(msg_type, "UNKNOWN")
            message = {"timestamp": timestamp, "type": msg_type_str}

            if msg_type == 1:
                body = f.read(12)
                if len(body) < 12: break
                order_id, volume, price = struct.unpack("IIf", body)
                message.update({"order_id": order_id, "volume": volume, "price": price})

            elif msg_type == 2:
                body = f.read(8)
                if len(body) < 8: break
                order_id, exec_qty = struct.unpack("II", body)
                message.update({"order_id": order_id, "exec_qty": exec_qty})

            elif msg_type == 3:
                body = f.read(4)
                if len(body) < 4: break
                order_id, = struct.unpack("I", body)
                message.update({"order_id": order_id})

            elif msg_type == 4:
                body = f.read(12)
                if len(body) < 12: break
                level, bid, ask = struct.unpack("Iff", body)
                message.update({"depth_level": level, "bid": bid, "ask": ask})

            elif msg_type == 5:
                body = f.read(1)
                if len(body) < 1: break
                status, = struct.unpack("B", body)
                status_str = ["CLOSED", "OPEN", "HALT"][status] if status < 3 else "UNKNOWN"
                message.update({"market_status": status_str})

            elif msg_type == 6:
                body = f.read(50)
                if len(body) < 50: break
                headline = body.rstrip(b'\x00').decode("utf-8", errors="ignore")
                message.update({"headline": headline})

            elif msg_type == 7:
                body = f.read(23)
                if len(body) < 23: break
                symbol, entry_type, price, size, position = struct.unpack("10sBfII", body)
                symbol = symbol.decode("utf-8").strip('\x00')
                side = "Bid" if entry_type == 0 else "Ask"
                message.update({
                    "symbol": symbol,
                    "side": side,
                    "price": price,
                    "size": size,
                    "position": position
                })

            messages.append(message)

    with open(output_file, "w") as f:
        json.dump(messages, f, indent=2)

    print(f"[✓] JSON file written to {output_file}")

if __name__ == "__main__":
    decode_log_to_json()

import struct
import csv
from datetime import datetime

# Message Types Mapping
MESSAGE_TYPES = {
    1: "ORDER_ADD",
    2: "ORDER_EXEC",
    3: "ORDER_CANCEL",
    4: "DEPTH_UPDATE",
    5: "MARKET_STATUS",
    6: "SPECIAL_NEWS"
}

def decode_log_file(input_file="fast_extended_10k.log", output_csv="decoded_fast_log.csv"):
    rows = []

    with open(input_file, "rb") as f:
        while True:
            # Read message type (1 byte) + timestamp (8 bytes)
            header = f.read(9)
            if not header or len(header) < 9:
                break

            msg_type, ts = struct.unpack("Bd", header)
            timestamp = datetime.fromtimestamp(ts).isoformat()
            msg_type_str = MESSAGE_TYPES.get(msg_type, "UNKNOWN")

            # Decode based on message type
            if msg_type == 1:  # ORDER_ADD
                body = f.read(12)
                if len(body) < 12: break
                order_id, volume, price = struct.unpack("IIf", body)
                print(f"{timestamp} | ORDER_ADD: ID={order_id}, Vol={volume}, Price={price}")
                rows.append([timestamp, msg_type_str, order_id, volume, price, "", "", "", ""])
            
            elif msg_type == 2:  # ORDER_EXEC
                body = f.read(8)
                if len(body) < 8: break
                order_id, exec_qty = struct.unpack("II", body)
                print(f"{timestamp} | ORDER_EXEC: ID={order_id}, Qty={exec_qty}")
                rows.append([timestamp, msg_type_str, order_id, "", "", exec_qty, "", "", ""])
            
            elif msg_type == 3:  # ORDER_CANCEL
                body = f.read(4)
                if len(body) < 4: break
                order_id, = struct.unpack("I", body)
                print(f"{timestamp} | ORDER_CANCEL: ID={order_id}")
                rows.append([timestamp, msg_type_str, order_id, "", "", "", "", "", ""])
            
            elif msg_type == 4:  # DEPTH_UPDATE
                body = f.read(12)
                if len(body) < 12: break
                level, bid, ask = struct.unpack("Iff", body)
                print(f"{timestamp} | DEPTH_UPDATE: Level={level}, Bid={bid}, Ask={ask}")
                rows.append([timestamp, msg_type_str, "", "", "", "", level, bid, ask])
            
            elif msg_type == 5:  # MARKET_STATUS
                body = f.read(1)
                if len(body) < 1: break
                status, = struct.unpack("B", body)
                status_str = ["CLOSED", "OPEN", "HALT"][status] if status < 3 else "UNKNOWN"
                print(f"{timestamp} | MARKET_STATUS: {status_str}")
                rows.append([timestamp, msg_type_str, "", "", "", "", "", "", status_str])
            
            elif msg_type == 6:  # SPECIAL_NEWS
                body = f.read(50)
                if len(body) < 50: break
                headline = body.rstrip(b'\x00').decode("utf-8", errors="ignore")
                print(f"{timestamp} | SPECIAL_NEWS: {headline}")
                rows.append([timestamp, msg_type_str, "", "", "", "", "", "", headline])

    # Write CSV
    with open(output_csv, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Timestamp", "MessageType", "OrderID", "Volume", "Price",
            "ExecQty", "DepthLevel", "Bid", "Ask/Status/Headline"
        ])
        writer.writerows(rows)
    
    print(f"\n[✓] Decoded messages saved to {output_csv}")

if __name__ == "__main__":
    decode_log_file()

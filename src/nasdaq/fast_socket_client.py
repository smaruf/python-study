import socket
import struct
from datetime import datetime

HOST = "localhost"
PORT = 9009

MESSAGE_TYPES = {
    1: "ORDER_ADD",
    2: "ORDER_EXEC",
    3: "ORDER_CANCEL",
    4: "DEPTH_UPDATE",
    5: "MARKET_STATUS",
    6: "SPECIAL_NEWS",
    7: "MD_BY_PRICE"
}

def decode_stream(conn):
    while True:
        header = conn.recv(9)
        if not header: break
        msg_type, ts = struct.unpack("Bd", header)
        timestamp = datetime.fromtimestamp(ts).isoformat()
        msg_type_str = MESSAGE_TYPES.get(msg_type, "UNKNOWN")
        print(f"\n[{timestamp}] {msg_type_str}:")

        if msg_type == 1:
            body = conn.recv(12)
            order_id, volume, price = struct.unpack("IIf", body)
            print(f"  Order ID: {order_id}, Volume: {volume}, Price: {price:.2f}")

        elif msg_type == 2:
            body = conn.recv(8)
            order_id, qty = struct.unpack("II", body)
            print(f"  Order ID: {order_id}, Exec Qty: {qty}")

        elif msg_type == 3:
            body = conn.recv(4)
            order_id, = struct.unpack("I", body)
            print(f"  Order ID: {order_id} (Cancelled)")

        elif msg_type == 4:
            body = conn.recv(12)
            level, bid, ask = struct.unpack("Iff", body)
            print(f"  Level: {level}, Bid: {bid:.2f}, Ask: {ask:.2f}")

        elif msg_type == 5:
            body = conn.recv(1)
            status, = struct.unpack("B", body)
            status_str = ["CLOSED", "OPEN", "HALT"][status] if status < 3 else "UNKNOWN"
            print(f"  Status: {status_str}")

        elif msg_type == 6:
            body = conn.recv(50)
            headline = body.rstrip(b'\x00').decode("utf-8", errors="ignore")
            print(f"  News: {headline}")

        elif msg_type == 7:
            body = conn.recv(23)
            symbol, entry_type, price, size, position = struct.unpack("10sBfII", body)
            symbol = symbol.decode("utf-8").strip('\x00')
            side = "Bid" if entry_type == 0 else "Ask"
            print(f"  {symbol} - {side} x {size} @ {price:.2f}, Position {position}")

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        print("[🔌] Connected to FAST stream server")
        decode_stream(s)

if __name__ == "__main__":
    main()

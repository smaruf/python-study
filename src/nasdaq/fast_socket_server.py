import socket
import struct
import random
import time

HOST = "localhost"
PORT = 9009

def generate_message():
    msg_type = random.randint(1, 7)
    timestamp = time.time()
    base = struct.pack("Bd", msg_type, timestamp)

    if msg_type == 1:  # ORDER_ADD
        return base + struct.pack("IIf", random.randint(10000, 99999), random.randint(10, 1000), random.uniform(50, 200))
    elif msg_type == 2:  # ORDER_EXEC
        return base + struct.pack("II", random.randint(10000, 99999), random.randint(1, 500))
    elif msg_type == 3:  # ORDER_CANCEL
        return base + struct.pack("I", random.randint(10000, 99999))
    elif msg_type == 4:  # DEPTH_UPDATE
        return base + struct.pack("Iff", random.randint(1, 5), random.uniform(90, 150), random.uniform(91, 151))
    elif msg_type == 5:  # MARKET_STATUS
        return base + struct.pack("B", random.randint(0, 2))
    elif msg_type == 6:  # SPECIAL_NEWS
        headline = random.choice(["NEWS", "HALT", "OPEN", "BULL", "BEAR"]).ljust(50, '\x00').encode("utf-8")
        return base + headline
    elif msg_type == 7:  # MD_BY_PRICE
        symbol = random.choice(["ACI", "BXPHARMA", "SQURPHARMA"]).encode("utf-8").ljust(10, b'\x00')
        return base + symbol + struct.pack("BfII", random.randint(0, 1), random.uniform(100, 200), random.randint(100, 10000), random.randint(1, 5))

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"[🟢] FAST Server listening on {HOST}:{PORT}")
        conn, addr = s.accept()
        with conn:
            print(f"[➡️] Connected by {addr}")
            while True:
                msg = generate_message()
                conn.sendall(msg)
                time.sleep(0.01)  # simulate tick frequency

if __name__ == "__main__":
    main()

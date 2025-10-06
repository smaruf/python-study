import socket
import struct
from collections import defaultdict

# --- Configuration: Update these with your actual multicast settings ---
MCAST_GRP = '239.255.0.1'  # Your exchange group
MCAST_PORT = 5000          # Your exchange port

# --- Order book data structures ---
class MarketDepth:
    def __init__(self):
        self.bids = defaultdict(lambda: {'qty': 0, 'orders': 0})
        self.asks = defaultdict(lambda: {'qty': 0, 'orders': 0})

    def update(self, price, qty, orders, side):
        book = self.bids if side == 'bid' else self.asks
        book[price] = {'qty': qty, 'orders': orders}

    def __str__(self):
        bid_levels = sorted(self.bids.items(), reverse=True)
        ask_levels = sorted(self.asks.items())
        s = "BIDS:\n"
        for price, data in bid_levels:
            s += f"  {price}: {data}\n"
        s += "ASKS:\n"
        for price, data in ask_levels:
            s += f"  {price}: {data}\n"
        return s

class OrderBook:
    def __init__(self):
        self.orders = {}

    def add_order(self, order_id, price, qty, side):
        self.orders[order_id] = {'price': price, 'qty': qty, 'side': side}

    def remove_order(self, order_id):
        if order_id in self.orders:
            del self.orders[order_id]

    def __str__(self):
        return str(self.orders)

# --- Stub decoder for demonstration; replace with real FAST decoder ---
def fast_decode(data):
    """
    Replace this stub with actual FAST decoder logic.
    For demo, returns mock messages as per CSE spec.
    """
    # Example: pretend we decoded a Market Data Incremental Refresh
    # Return a list of dict entries simulating parsed data
    return [
        {'type': 'depth', 'side': 'bid', 'price': 101.0, 'qty': 300, 'orders': 3},
        {'type': 'depth', 'side': 'ask', 'price': 102.0, 'qty': 200, 'orders': 2},
        {'type': 'order', 'order_id': 'A123', 'price': 101.0, 'qty': 100, 'side': 'bid', 'action': 'add'},
        {'type': 'order', 'order_id': 'B456', 'price': 102.0, 'qty': 200, 'side': 'ask', 'action': 'add'},
    ]

def main():
    market_depth = MarketDepth()
    order_book = OrderBook()

    # --- UDP Multicast setup ---
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('', MCAST_PORT))
    mreq = struct.pack("=4sl", socket.inet_aton(MCAST_GRP), socket.INADDR_ANY)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    print(f"Listening to FAST UDP feed at {MCAST_GRP}:{MCAST_PORT}")

    while True:
        data, addr = sock.recvfrom(65536)
        print(f"\nReceived {len(data)} bytes from {addr}")

        # --- Decode FAST message ---
        messages = fast_decode(data)  # Replace with actual decoder!

        # --- Process decoded data ---
        for msg in messages:
            if msg['type'] == 'depth':
                market_depth.update(msg['price'], msg['qty'], msg['orders'], msg['side'])
            elif msg['type'] == 'order':
                if msg['action'] == 'add':
                    order_book.add_order(msg['order_id'], msg['price'], msg['qty'], msg['side'])
                elif msg['action'] == 'remove':
                    order_book.remove_order(msg['order_id'])

        # --- Print book states ---
        print("\nMarket Depth:")
        print(market_depth)
        print("Order Book:")
        print(order_book)

if __name__ == "__main__":
    main()

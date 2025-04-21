import struct
import csv
import json

# Constants for message types
MESSAGE_TYPES = {
    1: "ORDER_ADD",
    2: "ORDER_EXEC",
    3: "ORDER_CANCEL",
    4: "DEPTH_UPDATE",
    5: "MARKET_STATUS",
    6: "SPECIAL_NEWS"
}

# Parse the given extended binary log file
def parse_extended_fast_log(file_name):
    parsed_data = []
    with open(file_name, "rb") as f:
        while chunk := f.read(9):  # Read type (1 byte) + timestamp (8 bytes)
            try:
                msg_type, timestamp = struct.unpack("Bd", chunk)
                msg_data = {"Message Type": MESSAGE_TYPES.get(msg_type, "UNKNOWN"), "Timestamp": timestamp}

                # Decode based on message type
                if msg_type == 1:  # ORDER_ADD
                    data_chunk = f.read(12)  # order_id (4 bytes), volume (4 bytes), price (4 bytes)
                    order_id, volume, price = struct.unpack("IIf", data_chunk)
                    msg_data.update({"Order ID": order_id, "Volume": volume, "Price": price})
                elif msg_type == 2:  # ORDER_EXEC
                    data_chunk = f.read(8)  # order_id (4 bytes), exec_qty (4 bytes)
                    order_id, exec_qty = struct.unpack("II", data_chunk)
                    msg_data.update({"Order ID": order_id, "Executed Quantity": exec_qty})
                elif msg_type == 3:  # ORDER_CANCEL
                    data_chunk = f.read(4)  # order_id (4 bytes)
                    order_id, = struct.unpack("I", data_chunk)
                    msg_data.update({"Order ID": order_id})
                elif msg_type == 4:  # DEPTH_UPDATE
                    data_chunk = f.read(12)  # depth_level (4 bytes), bid (4 bytes), ask (4 bytes)
                    depth_level, bid, ask = struct.unpack("Iff", data_chunk)
                    msg_data.update({"Depth Level": depth_level, "Bid": bid, "Ask": ask})
                elif msg_type == 5:  # MARKET_STATUS
                    data_chunk = f.read(1)  # status (1 byte)
                    status, = struct.unpack("B", data_chunk)
                    msg_data.update({"Market Status": status})
                elif msg_type == 6:  # SPECIAL_NEWS
                    data_chunk = f.read(50)  # headline (50 bytes)
                    headline = struct.unpack("50s", data_chunk)[0].decode("utf-8").strip()
                    msg_data.update({"Headline": headline})
                else:
                    print(f"Unknown message type {msg_type}. Skipping...")

                parsed_data.append(msg_data)
            except struct.error:
                print("Incomplete or malformed record encountered. Skipping...")
    return parsed_data

# Save the parsed data to a CSV file
def save_to_csv(file_name, data):
    with open(file_name, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

# Save the parsed data to a JSON file
def save_to_json(file_name, data):
    with open(file_name, "w") as file:
        json.dump(data, file, indent=4)

if __name__ == "__main__":
    # Input binary log file (provide the FAST extended log file path)
    binary_log_file = "fast_extended_10k.log"
    
    # Output files
    csv_output = "extended_nasdaq_data.csv"
    json_output = "extended_nasdaq_data.json"
    
    # Parse the binary log file
    parsed_data = parse_extended_fast_log(binary_log_file)
    
    # Save the parsed data
    if parsed_data:
        save_to_csv(csv_output, parsed_data)
        save_to_json(json_output, parsed_data)
        print(f"Parsed data saved to {csv_output} and {json_output}")
    else:
        print("No valid data found in the log file.")

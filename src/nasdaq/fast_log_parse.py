import struct
import csv
import json

# Parse the given binary log file
def parse_fix_fast_log(file_name):
    parsed_data = []
    with open(file_name, "rb") as f:
        while chunk := f.read(38):  # Adjust the chunk size as per the FIX/FAST message size
            try:
                # Decode the binary data according to FIX/FAST schema (example fields)
                stock, timestamp, price, volume = struct.unpack("10s20sfI", chunk)
                parsed_data.append({
                    "Stock Symbol": stock.decode("utf-8").strip(),
                    "Timestamp": timestamp.decode("utf-8").strip(),
                    "Price": price,
                    "Volume": volume
                })
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
    # Input binary log file (provide the NASDAQ sample log file path)
    binary_log_file = "nasdaq_log_sample.bin"
    
    # Output files
    csv_output = "nasdaq_data.csv"
    json_output = "nasdaq_data.json"
    
    # Parse the binary log file
    parsed_data = parse_fix_fast_log(binary_log_file)
    
    # Save the parsed data
    if parsed_data:
        save_to_csv(csv_output, parsed_data)
        save_to_json(json_output, parsed_data)
        print(f"Parsed data saved to {csv_output} and {json_output}")
    else:
        print("No valid data found in the log file.")

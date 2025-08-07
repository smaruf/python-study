//// running in bash: c3c decode_fast_log_extended.c3 -o decode_log 
import std.io;
import std.fs;
import std.time;
import std.conv;
import std.string;

// Message type names
const char *MESSAGE_TYPES[7] = {
    "UNKNOWN",
    "ORDER_ADD",
    "ORDER_EXEC",
    "ORDER_CANCEL",
    "DEPTH_UPDATE",
    "MARKET_STATUS",
    "SPECIAL_NEWS"
};

void decode_log_file(const char *input_file, const char *output_csv) {
    File infile = File.open(input_file, "rb");
    File outfile = File.open(output_csv, "w");

    outfile.write("Timestamp,MessageType,OrderID,Volume,Price,ExecQty,DepthLevel,Bid,Ask/Status/Headline\n");

    while (true) {
        // Read header: 1 byte type + 8 bytes timestamp (double)
        uint8 msg_type;
        double timestamp_val;
        if (infile.read(&msg_type, 1) != 1) break;
        if (infile.read(&timestamp_val, 8) != 8) break;

        // Format timestamp
        string timestamp = std.time.strftime("%Y-%m-%dT%H:%M:%S", std.time.from_unix_time(timestamp_val));

        const char *msg_type_str = MESSAGE_TYPES[msg_type < 7 ? msg_type : 0];

        if (msg_type == 1) { // ORDER_ADD
            uint32 order_id, volume;
            float price;
            infile.read(&order_id, 4);
            infile.read(&volume, 4);
            infile.read(&price, 4);
            printf("%s | ORDER_ADD: ID=%u, Vol=%u, Price=%.2f\n", timestamp, order_id, volume, price);
            outfile.write("%s,%s,%u,%u,%.2f,,,,\n", timestamp, msg_type_str, order_id, volume, price);
        }
        else if (msg_type == 2) { // ORDER_EXEC
            uint32 order_id, exec_qty;
            infile.read(&order_id, 4);
            infile.read(&exec_qty, 4);
            printf("%s | ORDER_EXEC: ID=%u, Qty=%u\n", timestamp, order_id, exec_qty);
            outfile.write("%s,%s,%u,,,%u,,,,\n", timestamp, msg_type_str, order_id, exec_qty);
        }
        else if (msg_type == 3) { // ORDER_CANCEL
            uint32 order_id;
            infile.read(&order_id, 4);
            printf("%s | ORDER_CANCEL: ID=%u\n", timestamp, order_id);
            outfile.write("%s,%s,%u,,,,,,,\n", timestamp, msg_type_str, order_id);
        }
        else if (msg_type == 4) { // DEPTH_UPDATE
            uint32 level;
            float bid, ask;
            infile.read(&level, 4);
            infile.read(&bid, 4);
            infile.read(&ask, 4);
            printf("%s | DEPTH_UPDATE: Level=%u, Bid=%.2f, Ask=%.2f\n", timestamp, level, bid, ask);
            outfile.write("%s,%s,,,,%u,%.2f,%.2f,\n", timestamp, msg_type_str, level, bid, ask);
        }
        else if (msg_type == 5) { // MARKET_STATUS
            uint8 status;
            infile.read(&status, 1);
            const char *status_str = status == 0 ? "CLOSED" : (status == 1 ? "OPEN" : (status == 2 ? "HALT" : "UNKNOWN"));
            printf("%s | MARKET_STATUS: %s\n", timestamp, status_str);
            outfile.write("%s,%s,,,,,,,%s\n", timestamp, msg_type_str, status_str);
        }
        else if (msg_type == 6) { // SPECIAL_NEWS
            uint8 headline_raw[50];
            infile.read(&headline_raw, 50);
            string headline = std.string.from_utf8(headline_raw, std.string.strnlen(headline_raw, 50));
            printf("%s | SPECIAL_NEWS: %s\n", timestamp, headline);
            outfile.write("%s,%s,,,,,,,%s\n", timestamp, msg_type_str, headline);
        }
        else {
            break;
        }
    }
    infile.close();
    outfile.close();
    printf("\n[✓] Decoded messages saved to %s\n", output_csv);
}

int main() {
    decode_log_file("fast_extended_10k.log", "decoded_fast_log.csv");
    return 0;
}

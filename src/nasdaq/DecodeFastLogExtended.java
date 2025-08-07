//// compile in Bash: javac DecodeFastLogExtended.java
//// then run in Bash: java DecodeFastLogExtended
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Date;

public class DecodeFastLogExtended {
    static final String[] MESSAGE_TYPES = {
        "UNKNOWN",
        "ORDER_ADD",
        "ORDER_EXEC",
        "ORDER_CANCEL",
        "DEPTH_UPDATE",
        "MARKET_STATUS",
        "SPECIAL_NEWS"
    };

    public static void decodeLogFile(String inputFile, String outputCsv) throws Exception {
        try (
            DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(inputFile)));
            PrintWriter out = new PrintWriter(new OutputStreamWriter(new FileOutputStream(outputCsv), StandardCharsets.UTF_8))
        ) {
            out.println("Timestamp,MessageType,OrderID,Volume,Price,ExecQty,DepthLevel,Bid,Ask/Status/Headline");

            while (in.available() > 0) {
                byte msgType = in.readByte();
                double timestampVal = in.readDouble();

                long seconds = (long) timestampVal;
                long millis = (long) ((timestampVal - seconds) * 1000);
                Date date = new Date(seconds * 1000 + millis);
                String timestampStr = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS").format(date);

                String msgTypeStr = msgType >= 0 && msgType < MESSAGE_TYPES.length ? MESSAGE_TYPES[msgType] : "UNKNOWN";

                if (msgType == 1) { // ORDER_ADD
                    long orderId = Integer.toUnsignedLong(in.readInt());
                    long volume = Integer.toUnsignedLong(in.readInt());
                    float price = in.readFloat();
                    System.out.printf("%s | ORDER_ADD: ID=%d, Vol=%d, Price=%.2f%n", timestampStr, orderId, volume, price);
                    out.printf("%s,%s,%d,%d,%.2f,,,,%n", timestampStr, msgTypeStr, orderId, volume, price);
                } else if (msgType == 2) { // ORDER_EXEC
                    long orderId = Integer.toUnsignedLong(in.readInt());
                    long execQty = Integer.toUnsignedLong(in.readInt());
                    System.out.printf("%s | ORDER_EXEC: ID=%d, Qty=%d%n", timestampStr, orderId, execQty);
                    out.printf("%s,%s,%d,,,%d,,,,%n", timestampStr, msgTypeStr, orderId, execQty);
                } else if (msgType == 3) { // ORDER_CANCEL
                    long orderId = Integer.toUnsignedLong(in.readInt());
                    System.out.printf("%s | ORDER_CANCEL: ID=%d%n", timestampStr, orderId);
                    out.printf("%s,%s,%d,,,,,,,%n", timestampStr, msgTypeStr, orderId);
                } else if (msgType == 4) { // DEPTH_UPDATE
                    long level = Integer.toUnsignedLong(in.readInt());
                    float bid = in.readFloat();
                    float ask = in.readFloat();
                    System.out.printf("%s | DEPTH_UPDATE: Level=%d, Bid=%.2f, Ask=%.2f%n", timestampStr, level, bid, ask);
                    out.printf("%s,%s,,,,%d,%.2f,%.2f,%n", timestampStr, msgTypeStr, level, bid, ask);
                } else if (msgType == 5) { // MARKET_STATUS
                    byte status = in.readByte();
                    String statusStr = status == 0 ? "CLOSED" : status == 1 ? "OPEN" : status == 2 ? "HALT" : "UNKNOWN";
                    System.out.printf("%s | MARKET_STATUS: %s%n", timestampStr, statusStr);
                    out.printf("%s,%s,,,,,,,%s%n", timestampStr, msgTypeStr, statusStr);
                } else if (msgType == 6) { // SPECIAL_NEWS
                    byte[] headlineBytes = new byte[50];
                    in.readFully(headlineBytes);
                    int headlineLen = 0;
                    for (; headlineLen < headlineBytes.length && headlineBytes[headlineLen] != 0; headlineLen++);
                    String headline = new String(headlineBytes, 0, headlineLen, StandardCharsets.UTF_8);
                    System.out.printf("%s | SPECIAL_NEWS: %s%n", timestampStr, headline);
                    out.printf("%s,%s,,,,,,,%s%n", timestampStr, msgTypeStr, headline.replace(",", " "));
                } else {
                    // Unknown type, break
                    break;
                }
            }
        }
        System.out.println("\n[✓] Decoded messages saved to decoded_fast_log.csv");
    }

    public static void main(String[] args) throws Exception {
        String inputFile = "fast_extended_10k.log";
        String outputCsv = "decoded_fast_log.csv";
        decodeLogFile(inputFile, outputCsv);
    }
}

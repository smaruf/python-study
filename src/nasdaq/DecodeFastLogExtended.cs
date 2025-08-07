//// running in bash: csc DecodeFastLogExtended.cs
using System;
using System.IO;
using System.Text;
using System.Globalization;

class DecodeFastLogExtended
{
    static readonly string[] MESSAGE_TYPES = {
        "UNKNOWN",
        "ORDER_ADD",
        "ORDER_EXEC",
        "ORDER_CANCEL",
        "DEPTH_UPDATE",
        "MARKET_STATUS",
        "SPECIAL_NEWS"
    };

    static void DecodeLogFile(string inputFile, string outputCsv)
    {
        using (var infile = new BinaryReader(File.Open(inputFile, FileMode.Open)))
        using (var outfile = new StreamWriter(outputCsv, false, Encoding.UTF8))
        {
            outfile.WriteLine("Timestamp,MessageType,OrderID,Volume,Price,ExecQty,DepthLevel,Bid,Ask/Status/Headline");

            while (infile.BaseStream.Position < infile.BaseStream.Length)
            {
                byte msgType = infile.ReadByte();
                double timestampVal = infile.ReadDouble();

                DateTime timestamp = DateTimeOffset.FromUnixTimeSeconds((long)timestampVal)
                    .UtcDateTime.AddSeconds(timestampVal - Math.Floor(timestampVal));
                string timestampStr = timestamp.ToString("yyyy-MM-ddTHH:mm:ss.fff", CultureInfo.InvariantCulture);

                string msgTypeStr = (msgType < MESSAGE_TYPES.Length) ? MESSAGE_TYPES[msgType] : "UNKNOWN";

                if (msgType == 1) // ORDER_ADD
                {
                    uint orderId = infile.ReadUInt32();
                    uint volume = infile.ReadUInt32();
                    float price = infile.ReadSingle();
                    Console.WriteLine($"{timestampStr} | ORDER_ADD: ID={orderId}, Vol={volume}, Price={price}");
                    outfile.WriteLine($"{timestampStr},{msgTypeStr},{orderId},{volume},{price},,,,,");
                }
                else if (msgType == 2) // ORDER_EXEC
                {
                    uint orderId = infile.ReadUInt32();
                    uint execQty = infile.ReadUInt32();
                    Console.WriteLine($"{timestampStr} | ORDER_EXEC: ID={orderId}, Qty={execQty}");
                    outfile.WriteLine($"{timestampStr},{msgTypeStr},{orderId},,,{execQty},,,,");
                }
                else if (msgType == 3) // ORDER_CANCEL
                {
                    uint orderId = infile.ReadUInt32();
                    Console.WriteLine($"{timestampStr} | ORDER_CANCEL: ID={orderId}");
                    outfile.WriteLine($"{timestampStr},{msgTypeStr},{orderId},,,,,,,");
                }
                else if (msgType == 4) // DEPTH_UPDATE
                {
                    uint level = infile.ReadUInt32();
                    float bid = infile.ReadSingle();
                    float ask = infile.ReadSingle();
                    Console.WriteLine($"{timestampStr} | DEPTH_UPDATE: Level={level}, Bid={bid}, Ask={ask}");
                    outfile.WriteLine($"{timestampStr},{msgTypeStr},,,,,{level},{bid},{ask},");
                }
                else if (msgType == 5) // MARKET_STATUS
                {
                    byte status = infile.ReadByte();
                    string statusStr = status switch
                    {
                        0 => "CLOSED",
                        1 => "OPEN",
                        2 => "HALT",
                        _ => "UNKNOWN"
                    };
                    Console.WriteLine($"{timestampStr} | MARKET_STATUS: {statusStr}");
                    outfile.WriteLine($"{timestampStr},{msgTypeStr},,,,,,,,{statusStr}");
                }
                else if (msgType == 6) // SPECIAL_NEWS
                {
                    byte[] headlineBytes = infile.ReadBytes(50);
                    int headlineLen = Array.IndexOf(headlineBytes, (byte)0);
                    if (headlineLen < 0) headlineLen = headlineBytes.Length;
                    string headline = Encoding.UTF8.GetString(headlineBytes, 0, headlineLen);
                    Console.WriteLine($"{timestampStr} | SPECIAL_NEWS: {headline}");
                    outfile.WriteLine($"{timestampStr},{msgTypeStr},,,,,,,,{headline}");
                }
                else
                {
                    // Unknown type, exit loop
                    break;
                }
            }
        }
        Console.WriteLine("\n[✓] Decoded messages saved to decoded_fast_log.csv");
    }

    static void Main(string[] args)
    {
        string inputFile = "fast_extended_10k.log";
        string outputCsv = "decoded_fast_log.csv";
        DecodeLogFile(inputFile, outputCsv);
    }
}

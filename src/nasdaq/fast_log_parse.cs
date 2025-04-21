
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using CsvHelper;
using System.Globalization;
using Newtonsoft.Json;

class Program
{
    // Constants for message types
    private static readonly Dictionary<int, string> MESSAGE_TYPES = new Dictionary<int, string>
    {
        { 1, "ORDER_ADD" },
        { 2, "ORDER_EXEC" },
        { 3, "ORDER_CANCEL" },
        { 4, "DEPTH_UPDATE" },
        { 5, "MARKET_STATUS" },
        { 6, "SPECIAL_NEWS" }
    };

    // Parse the given extended binary log file
    public static List<Dictionary<string, object>> ParseExtendedFastLog(string fileName)
    {
        var parsedData = new List<Dictionary<string, object>>();

        using (var fileStream = new FileStream(fileName, FileMode.Open, FileAccess.Read))
        using (var reader = new BinaryReader(fileStream))
        {
            while (reader.BaseStream.Position != reader.BaseStream.Length)
            {
                try
                {
                    var msgType = reader.ReadByte();
                    var timestamp = reader.ReadDouble();

                    var msgData = new Dictionary<string, object>
                    {
                        { "Message Type", MESSAGE_TYPES.ContainsKey(msgType) ? MESSAGE_TYPES[msgType] : "UNKNOWN" },
                        { "Timestamp", timestamp }
                    };

                    // Decode based on message type
                    switch (msgType)
                    {
                        case 1: // ORDER_ADD
                            var orderId1 = reader.ReadUInt32();
                            var volume = reader.ReadUInt32();
                            var price = reader.ReadSingle();
                            msgData.Add("Order ID", orderId1);
                            msgData.Add("Volume", volume);
                            msgData.Add("Price", price);
                            break;
                        case 2: // ORDER_EXEC
                            var orderId2 = reader.ReadUInt32();
                            var execQty = reader.ReadUInt32();
                            msgData.Add("Order ID", orderId2);
                            msgData.Add("Executed Quantity", execQty);
                            break;
                        case 3: // ORDER_CANCEL
                            var orderId3 = reader.ReadUInt32();
                            msgData.Add("Order ID", orderId3);
                            break;
                        case 4: // DEPTH_UPDATE
                            var depthLevel = reader.ReadUInt32();
                            var bid = reader.ReadSingle();
                            var ask = reader.ReadSingle();
                            msgData.Add("Depth Level", depthLevel);
                            msgData.Add("Bid", bid);
                            msgData.Add("Ask", ask);
                            break;
                        case 5: // MARKET_STATUS
                            var status = reader.ReadByte();
                            msgData.Add("Market Status", status);
                            break;
                        case 6: // SPECIAL_NEWS
                            var headlineBytes = reader.ReadBytes(50);
                            var headline = Encoding.UTF8.GetString(headlineBytes).TrimEnd('\0');
                            msgData.Add("Headline", headline);
                            break;
                        default:
                            Console.WriteLine($"Unknown message type {msgType}. Skipping...");
                            continue;
                    }

                    parsedData.Add(msgData);
                }
                catch (EndOfStreamException)
                {
                    Console.WriteLine("Incomplete or malformed record encountered. Skipping...");
                    break;
                }
            }
        }

        return parsedData;
    }

    // Save the parsed data to a CSV file
    public static void SaveToCsv(string fileName, List<Dictionary<string, object>> data)
    {
        using (var writer = new StreamWriter(fileName))
        using (var csv = new CsvHelper.CsvWriter(writer, CultureInfo.InvariantCulture))
        {
            var headers = new List<string>(data[0].Keys);
            foreach (var header in headers)
            {
                csv.WriteField(header);
            }
            csv.NextRecord();

            foreach (var record in data)
            {
                foreach (var header in headers)
                {
                    csv.WriteField(record[header]);
                }
                csv.NextRecord();
            }
        }
    }

    // Save the parsed data to a JSON file
    public static void SaveToJson(string fileName, List<Dictionary<string, object>> data)
    {
        var json = JsonConvert.SerializeObject(data, Formatting.Indented);
        File.WriteAllText(fileName, json);
    }

    static void Main(string[] args)
    {
        // Input binary log file (provide the FAST extended log file path)
        string binaryLogFile = "fast_extended_10k.log";

        // Output files
        string csvOutput = "extended_nasdaq_data.csv";
        string jsonOutput = "extended_nasdaq_data.json";

        // Parse the binary log file
        var parsedData = ParseExtendedFastLog(binaryLogFile);

        // Save the parsed data
        if (parsedData.Count > 0)
        {
            SaveToCsv(csvOutput, parsedData);
            SaveToJson(jsonOutput, parsedData);
            Console.WriteLine($"Parsed data saved to {csvOutput} and {jsonOutput}");
        }
        else
        {
            Console.WriteLine("No valid data found in the log file.");
        }
    }
}

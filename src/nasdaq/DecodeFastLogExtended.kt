/// compile in bash: kotlinc DecodeFastLogExtended.kt -include-runtime -d DecodeFastLogExtended.jar
/// run in bash: java -jar DecodeFastLogExtended.jar
import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.Charset
import java.text.SimpleDateFormat
import java.util.*

val MESSAGE_TYPES = arrayOf(
    "UNKNOWN",
    "ORDER_ADD",
    "ORDER_EXEC",
    "ORDER_CANCEL",
    "DEPTH_UPDATE",
    "MARKET_STATUS",
    "SPECIAL_NEWS"
)

fun main() {
    val inputFile = "fast_extended_10k.log"
    val outputCsv = "decoded_fast_log.csv"
    decodeLogFile(inputFile, outputCsv)
}

fun decodeLogFile(inputFile: String, outputCsv: String) {
    DataInputStream(BufferedInputStream(FileInputStream(inputFile))).use { infile ->
        PrintWriter(OutputStreamWriter(FileOutputStream(outputCsv), Charset.forName("UTF-8"))).use { outfile ->
            outfile.println("Timestamp,MessageType,OrderID,Volume,Price,ExecQty,DepthLevel,Bid,Ask/Status/Headline")

            while (infile.available() > 0) {
                val msgType = infile.readUnsignedByte()
                val timestampVal = infile.readDouble()
                val seconds = timestampVal.toLong()
                val millis = ((timestampVal - seconds) * 1000).toLong()
                val date = Date(seconds * 1000 + millis)
                val timestampStr = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS").format(date)
                val msgTypeStr = if (msgType in MESSAGE_TYPES.indices) MESSAGE_TYPES[msgType] else "UNKNOWN"

                when (msgType) {
                    1 -> { // ORDER_ADD
                        val orderId = infile.readInt().toUInt()
                        val volume = infile.readInt().toUInt()
                        val price = infile.readFloat()
                        println("$timestampStr | ORDER_ADD: ID=$orderId, Vol=$volume, Price=$price")
                        outfile.println("$timestampStr,$msgTypeStr,$orderId,$volume,$price,,,,")
                    }
                    2 -> { // ORDER_EXEC
                        val orderId = infile.readInt().toUInt()
                        val execQty = infile.readInt().toUInt()
                        println("$timestampStr | ORDER_EXEC: ID=$orderId, Qty=$execQty")
                        outfile.println("$timestampStr,$msgTypeStr,$orderId,,,$execQty,,,,")
                    }
                    3 -> { // ORDER_CANCEL
                        val orderId = infile.readInt().toUInt()
                        println("$timestampStr | ORDER_CANCEL: ID=$orderId")
                        outfile.println("$timestampStr,$msgTypeStr,$orderId,,,,,,,")
                    }
                    4 -> { // DEPTH_UPDATE
                        val level = infile.readInt().toUInt()
                        val bid = infile.readFloat()
                        val ask = infile.readFloat()
                        println("$timestampStr | DEPTH_UPDATE: Level=$level, Bid=$bid, Ask=$ask")
                        outfile.println("$timestampStr,$msgTypeStr,,,,$level,$bid,$ask,")
                    }
                    5 -> { // MARKET_STATUS
                        val status = infile.readUnsignedByte()
                        val statusStr = when (status) {
                            0 -> "CLOSED"
                            1 -> "OPEN"
                            2 -> "HALT"
                            else -> "UNKNOWN"
                        }
                        println("$timestampStr | MARKET_STATUS: $statusStr")
                        outfile.println("$timestampStr,$msgTypeStr,,,,,,,$statusStr")
                    }
                    6 -> { // SPECIAL_NEWS
                        val headlineBytes = ByteArray(50)
                        infile.readFully(headlineBytes)
                        val headlineLen = headlineBytes.indexOf(0).let { if (it < 0) headlineBytes.size else it }
                        val headline = String(headlineBytes, 0, headlineLen, Charset.forName("UTF-8"))
                        println("$timestampStr | SPECIAL_NEWS: $headline")
                        outfile.println("$timestampStr,$msgTypeStr,,,,,,,$headline")
                    }
                    else -> break
                }
            }
        }
    }
    println("\n[✓] Decoded messages saved to $outputCsv")
}

/// build in bash: go build decode_fast_log_extended.go
/// run in bash: ./decode_fast_log_extended
package main

import (
	"encoding/binary"
	"encoding/csv"
	"fmt"
	"os"
	"time"
)

var messageTypes = []string{
	"UNKNOWN",
	"ORDER_ADD",
	"ORDER_EXEC",
	"ORDER_CANCEL",
	"DEPTH_UPDATE",
	"MARKET_STATUS",
	"SPECIAL_NEWS",
}

func main() {
	inputFile := "fast_extended_10k.log"
	outputFile := "decoded_fast_log.csv"

	infile, err := os.Open(inputFile)
	if err != nil {
		panic(err)
	}
	defer infile.Close()

	outfile, err := os.Create(outputFile)
	if err != nil {
		panic(err)
	}
	defer outfile.Close()

	writer := csv.NewWriter(outfile)
	defer writer.Flush()

	writer.Write([]string{"Timestamp", "MessageType", "OrderID", "Volume", "Price", "ExecQty", "DepthLevel", "Bid", "Ask/Status/Headline"})

	for {
		header := make([]byte, 9)
		if _, err := infile.Read(header); err != nil {
			break
		}
		msgType := header[0]
		tsBits := binary.LittleEndian.Uint64(header[1:9])
		ts := float64(tsBits)
		secs := int64(ts)
		nsecs := int64((ts - float64(secs)) * 1e9)
		timestamp := time.Unix(secs, nsecs).UTC().Format("2006-01-02T15:04:05.000")

		msgTypeStr := "UNKNOWN"
		if int(msgType) < len(messageTypes) {
			msgTypeStr = messageTypes[msgType]
		}

		switch msgType {
		case 1: // ORDER_ADD
			body := make([]byte, 12)
			infile.Read(body)
			orderID := binary.LittleEndian.Uint32(body[0:4])
			volume := binary.LittleEndian.Uint32(body[4:8])
			price := binary.LittleEndian.Uint32(body[8:12])
			priceF := float32(price)
			fmt.Printf("%s | ORDER_ADD: ID=%d, Vol=%d, Price=%.2f\n", timestamp, orderID, volume, priceF)
			writer.Write([]string{timestamp, msgTypeStr, fmt.Sprint(orderID), fmt.Sprint(volume), fmt.Sprintf("%.2f", priceF), "", "", "", ""})
		case 2: // ORDER_EXEC
			body := make([]byte, 8)
			infile.Read(body)
			orderID := binary.LittleEndian.Uint32(body[0:4])
			execQty := binary.LittleEndian.Uint32(body[4:8])
			fmt.Printf("%s | ORDER_EXEC: ID=%d, Qty=%d\n", timestamp, orderID, execQty)
			writer.Write([]string{timestamp, msgTypeStr, fmt.Sprint(orderID), "", "", fmt.Sprint(execQty), "", "", ""})
		case 3: // ORDER_CANCEL
			body := make([]byte, 4)
			infile.Read(body)
			orderID := binary.LittleEndian.Uint32(body)
			fmt.Printf("%s | ORDER_CANCEL: ID=%d\n", timestamp, orderID)
			writer.Write([]string{timestamp, msgTypeStr, fmt.Sprint(orderID), "", "", "", "", "", ""})
		case 4: // DEPTH_UPDATE
			body := make([]byte, 12)
			infile.Read(body)
			level := binary.LittleEndian.Uint32(body[0:4])
			bid := binary.LittleEndian.Uint32(body[4:8])
			ask := binary.LittleEndian.Uint32(body[8:12])
			bidF := float32(bid)
			askF := float32(ask)
			fmt.Printf("%s | DEPTH_UPDATE: Level=%d, Bid=%.2f, Ask=%.2f\n", timestamp, level, bidF, askF)
			writer.Write([]string{timestamp, msgTypeStr, "", "", "", "", fmt.Sprint(level), fmt.Sprintf("%.2f", bidF), fmt.Sprintf("%.2f", askF)})
		case 5: // MARKET_STATUS
			body := make([]byte, 1)
			infile.Read(body)
			status := body[0]
			statusStr := "UNKNOWN"
			switch status {
			case 0:
				statusStr = "CLOSED"
			case 1:
				statusStr = "OPEN"
			case 2:
				statusStr = "HALT"
			}
			fmt.Printf("%s | MARKET_STATUS: %s\n", timestamp, statusStr)
			writer.Write([]string{timestamp, msgTypeStr, "", "", "", "", "", "", statusStr})
		case 6: // SPECIAL_NEWS
			body := make([]byte, 50)
			infile.Read(body)
			headline := ""
			for i := 0; i < 50; i++ {
				if body[i] == 0 {
					break
				}
				headline += string(body[i])
			}
			fmt.Printf("%s | SPECIAL_NEWS: %s\n", timestamp, headline)
			writer.Write([]string{timestamp, msgTypeStr, "", "", "", "", "", "", headline})
		default:
			break
		}
	}
	fmt.Printf("\n[✓] Decoded messages saved to %s\n", outputFile)
}

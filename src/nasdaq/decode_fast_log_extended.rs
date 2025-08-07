use std::fs::File;
use std::io::{BufReader, Read};
use std::error::Error;
use chrono::{NaiveDateTime, Utc, DateTime};
use csv::WriterBuilder;

const MESSAGE_TYPES: [&str; 7] = [
    "UNKNOWN", "ORDER_ADD", "ORDER_EXEC", "ORDER_CANCEL",
    "DEPTH_UPDATE", "MARKET_STATUS", "SPECIAL_NEWS",
];

fn decode_log_file(input_file: &str, output_csv: &str) -> Result<(), Box<dyn Error>> {
    let file = File::open(input_file)?;
    let mut reader = BufReader::new(file);

    let mut wtr = WriterBuilder::new().has_headers(true).from_path(output_csv)?;
    wtr.write_record(&[
        "Timestamp", "MessageType", "OrderID", "Volume", "Price",
        "ExecQty", "DepthLevel", "Bid", "Ask/Status/Headline"
    ])?;

    loop {
        let mut header = [0u8; 9];
        if reader.read_exact(&mut header).is_err() {
            break;
        }
        let msg_type = header[0];
        let ts = f64::from_le_bytes(header[1..9].try_into().unwrap());
        let secs = ts as i64;
        let nsecs = ((ts.fract() * 1e9) as u32);
        let timestamp = DateTime::<Utc>::from_utc(
            NaiveDateTime::from_timestamp_opt(secs, nsecs).unwrap(),
            Utc
        ).to_rfc3339();

        let msg_type_str = MESSAGE_TYPES.get(msg_type as usize).unwrap_or(&"UNKNOWN");

        match msg_type {
            1 => { // ORDER_ADD
                let mut body = [0u8; 12];
                if reader.read_exact(&mut body).is_err() { break; }
                let order_id = u32::from_le_bytes(body[0..4].try_into().unwrap());
                let volume = u32::from_le_bytes(body[4..8].try_into().unwrap());
                let price = f32::from_le_bytes(body[8..12].try_into().unwrap());
                println!("{timestamp} | ORDER_ADD: ID={order_id}, Vol={volume}, Price={price}");
                wtr.write_record(&[
                    &timestamp, msg_type_str, &order_id.to_string(),
                    &volume.to_string(), &price.to_string(),
                    "", "", "", ""
                ])?;
            },
            2 => { // ORDER_EXEC
                let mut body = [0u8; 8];
                if reader.read_exact(&mut body).is_err() { break; }
                let order_id = u32::from_le_bytes(body[0..4].try_into().unwrap());
                let exec_qty = u32::from_le_bytes(body[4..8].try_into().unwrap());
                println!("{timestamp} | ORDER_EXEC: ID={order_id}, Qty={exec_qty}");
                wtr.write_record(&[
                    &timestamp, msg_type_str, &order_id.to_string(),
                    "", "", &exec_qty.to_string(), "", "", ""
                ])?;
            },
            3 => { // ORDER_CANCEL
                let mut body = [0u8; 4];
                if reader.read_exact(&mut body).is_err() { break; }
                let order_id = u32::from_le_bytes(body.try_into().unwrap());
                println!("{timestamp} | ORDER_CANCEL: ID={order_id}");
                wtr.write_record(&[
                    &timestamp, msg_type_str, &order_id.to_string(),
                    "", "", "", "", "", ""
                ])?;
            },
            4 => { // DEPTH_UPDATE
                let mut body = [0u8; 12];
                if reader.read_exact(&mut body).is_err() { break; }
                let level = u32::from_le_bytes(body[0..4].try_into().unwrap());
                let bid = f32::from_le_bytes(body[4..8].try_into().unwrap());
                let ask = f32::from_le_bytes(body[8..12].try_into().unwrap());
                println!("{timestamp} | DEPTH_UPDATE: Level={level}, Bid={bid}, Ask={ask}");
                wtr.write_record(&[
                    &timestamp, msg_type_str, "", "", "",
                    "", &level.to_string(), &bid.to_string(), &ask.to_string()
                ])?;
            },
            5 => { // MARKET_STATUS
                let mut body = [0u8; 1];
                if reader.read_exact(&mut body).is_err() { break; }
                let status = body[0] as usize;
                let status_str = ["CLOSED", "OPEN", "HALT"]
                    .get(status).unwrap_or(&"UNKNOWN");
                println!("{timestamp} | MARKET_STATUS: {status_str}");
                wtr.write_record(&[
                    &timestamp, msg_type_str, "", "", "",
                    "", "", "", status_str
                ])?;
            },
            6 => { // SPECIAL_NEWS
                let mut body = [0u8; 50];
                if reader.read_exact(&mut body).is_err() { break; }
                let headline = String::from_utf8(
                    body.iter().cloned().take_while(|&b| b != 0).collect()
                ).unwrap_or_default();
                println!("{timestamp} | SPECIAL_NEWS: {headline}");
                wtr.write_record(&[
                    &timestamp, msg_type_str, "", "", "",
                    "", "", "", &headline
                ])?;
            },
            _ => break,
        }
    }

    wtr.flush()?;
    println!("\n[✓] Decoded messages saved to {output_csv}");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    // Change these file names as needed
    decode_log_file("fast_extended_10k.log", "decoded_fast_log.csv")
}

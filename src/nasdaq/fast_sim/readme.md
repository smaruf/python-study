## Simulator using Kafka and WebUI

```bash

fast_sim/
├── fast_kafka_producer.py        # Encoder + Kafka producer
├── fast_kafka_consumer_web.py    # Decoder + Kafka consumer + web server
├── fast_template.py              # Shared template for encoding/decoding
└── web_ui/
    ├── index.html                # Live feed dashboard
    └── script.js                 # WebSocket client
```

## FIX simulator with Kafka and WebUI

```bash
fix_sim/
├── fix_kafka_producer.py          # Random FIX message generator + producer
├── fix_kafka_consumer_web.py     # Kafka consumer + decoder + web server
├── fix_template.py                # Shared FIX message builder/parser
└── web_ui/
    ├── index.html
    └── script.js

```

from confluent_kafka import Consumer
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from fast_template import decode_message
import eventlet
eventlet.monkey_patch()

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

consumer = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'fast_group',
    'auto.offset.reset': 'latest'
})
consumer.subscribe(['fast_topic'])

@app.route("/")
def index():
    return render_template("index.html")

def consume_loop():
    while True:
        msg = consumer.poll(1.0)
        if msg is None or msg.error(): continue
        decoded = decode_message(msg.value())
        if decoded:
            print(f"[📥] {decoded}")
            socketio.emit('fast_data', decoded)

@socketio.on('connect')
def handle_connect():
    print("[🌐] Web client connected")

if __name__ == '__main__':
    print("[🔥] Starting FAST Web Server")
    socketio.start_background_task(consume_loop)
    socketio.run(app, host="0.0.0.0", port=5000)

from confluent_kafka import Consumer
from flask import Flask, render_template
from flask_socketio import SocketIO
from fix_template import parse_fix_message
import eventlet
eventlet.monkey_patch()

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

consumer = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'fix_group',
    'auto.offset.reset': 'latest'
})
consumer.subscribe(['fix_topic'])

@app.route("/")
def index():
    return render_template("index.html")

def consume_loop():
    while True:
        msg = consumer.poll(1.0)
        if msg is None or msg.error(): continue
        decoded = parse_fix_message(msg.value())
        socketio.emit('fix_data', decoded)

@socketio.on('connect')
def on_connect():
    print("[🌍] Web client connected")

if __name__ == "__main__":
    socketio.start_background_task(consume_loop)
    socketio.run(app, host="0.0.0.0", port=6000)

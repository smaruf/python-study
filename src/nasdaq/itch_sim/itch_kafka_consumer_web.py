from confluent_kafka import Consumer
from flask import Flask, render_template
from flask_socketio import SocketIO
from itch_template import decode_itch
import eventlet
eventlet.monkey_patch()

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

consumer = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'itch_group',
    'auto.offset.reset': 'latest'
})
consumer.subscribe(['itch_topic'])

@app.route("/")
def index():
    return render_template("index.html")

def consume_loop():
    while True:
        msg = consumer.poll(1.0)
        if msg is None or msg.error(): continue
        decoded = decode_itch(msg.value())
        socketio.emit('itch_data', decoded)

@socketio.on('connect')
def on_connect():
    print("[🌐] Web client connected")

if __name__ == "__main__":
    socketio.start_background_task(consume_loop)
    socketio.run(app, host="0.0.0.0", port=7000)

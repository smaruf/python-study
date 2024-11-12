import threading

class AtomicCounter:
    def __init__(self):
        self.value = 0
        self._lock = threading.Lock()

    def increment(self):
        with self._lock:
            self.value += 1  # Ensuring atomic increment

counter = AtomicCounter()

def increment_counter():
    for _ in range(10000):
        counter.increment()

threads = [threading.Thread(target=increment_counter) for _ in range(5)]
[t.start() for t in threads]
[t.join() for t in threads]

print(f"Counter value should be 50000, and it is: {counter.value}")

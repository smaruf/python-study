import feedparser
import requests
import ssl
import time

from functools import lru_cache, wraps
from datetime import datetime, timedelta

if hasattr(ssl, "_create_unverified_context"):
    ssl._create_default_https_context = ssl._create_unverified_context

def timed_lru_cache(seconds: int, maxsize: int = 128):
    def wrapper_cache(func):
        func = lru_cache(maxsize=maxsize)(func)
        func.lifetime = timedelta(seconds=seconds)
        func.expiration = datetime.utcnow() + func.lifetime

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if datetime.utcnow() >= func.expiration:
                func.cache_clear()
                func.expiration = datetime.utcnow() + func.lifetime

            return func(*args, **kwargs)

        return wrapped_func

    return wrapper_cache

@timed_lru_cache(10)
def get_article_from_server(url):
    print("Fetching article from server...")
    response = requests.get(url)
    return response.text

def monitor(url):
    maxlen = 45
    while True:
        print("\nChecking feed...")
        feed = feedparser.parse(url)

        for entry in feed.entries[:5]:
            if "python" in entry.title.lower():
                truncated_title = (
                    entry.title[:maxlen] + "..."
                    if len(entry.title) > maxlen
                    else entry.title
                )
                print(
                    "Match found:",
                    truncated_title,
                    len(get_article_from_server(entry.link)),
                )

        time.sleep(5)

monitor("https://realpython.com/atom.xml")

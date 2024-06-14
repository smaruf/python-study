import feedparser
import requests
import ssl
import time

if hasattr(ssl, "_create_unverified_context"):
    ssl._create_default_https_context = ssl._create_unverified_context

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

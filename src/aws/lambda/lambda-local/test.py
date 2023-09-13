from __future__ import print_function
from rx import Observable


def handler(event, context):
    xs = Observable.from_(range(event['answer']))
    ys = xs.to_blocking()
    zs = (x*x for x in ys if x % 7 == 0)
    for x in zs:
        print(x)

from functools import lru_cache
from time import time

class FunctionCacher:
  def __init__(self, func, ttl_seconds=3600):
    self.ttl_seconds = ttl_seconds
    self.func = func

  @lru_cache(maxsize=None)
  def _cached_func(self, *args, ttl_hash=None):
    del ttl_hash # if argument of function is same as before, then return cached, else load computation. that's why we manipulate argument.
    return self.func(*args)

  def _get_ttl_hash(self):
    """Return the same value within `ttl_seconds` time period"""
    return time() // self.ttl_seconds

  def __call__(self, *args):
    return self._cached_func(*args, ttl_hash=self._get_ttl_hash())

def func_without_cache(x, y):
  from time import sleep
  sleep(1)
  return x+y

func_with_cache = FunctionCacher(func_without_cache, ttl_seconds=1000)

_ = time()
print("\nPerform heavy computational task...")
print(func_with_cache(1,2), time() - _)

_ = time()
print("\nPerform same heavy computational task...")
print(func_with_cache(1,2), time() - _)

_ = time()
print("\nPerform different heavy computational task...")
print(func_with_cache(1,3), time() - _)

_ = time()
print("\nPerform previous heavy computational task...")
print(func_with_cache(1,3), time() - _)

# doc-link: https://stackoverflow.com/questions/31771286/python-in-memory-cache-with-time-to-live

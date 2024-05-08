import random
from hashlib import blake2b

def hash_it(word, length=16):
    if length % 2:
        raise ValueError('Length must be even')

    data = word.encode()
    digest_size = length // 2

    return blake2b(data, digest_size=digest_size).hexdigest()

class Backoff:
    _backoff_factor = 2

    def __init__(self, backoff, fuzz=0):
        self.backoff = backoff
        self.fuzz = fuzz

    def __iter__(self):
        backoff = self.backoff
        while True:
            if self.fuzz:
                backoff += backoff * random.uniform(-self.fuzz, self.fuzz)
            yield backoff

            backoff *= self._backoff_factor

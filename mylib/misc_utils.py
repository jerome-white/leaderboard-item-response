import random
import hashlib

def hash_it(word, length=16):
    if length % 2:
        raise ValueError('Length must be even')

    data = word.encode()
    digest_size = length // 2

    return (hashlib
            .blake2b(data, digest_size=digest_size)
            .hexdigest())

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

class FileChecksum:
    _method = 'md5'

    def __init__(self, data):
        self.data = data
        while data.suffix:
            data = data.with_suffix('')
        self.checksum = data.with_suffix(f'.{self._method}')

    def __str__(self):
        with self.data.open('rb') as fp:
            digest = hashlib.file_digest(fp, self._method)

        return digest.hexdigest()

    def __bool__(self):
        if not self.checksum.exists():
            return False

        return str(self) == self.checksum.read_text().strip()

    def write(self):
        with self.checksum.open('w') as fp:
            print(self, file=fp)

import hashlib
import struct
import random

def sha1_32bit(data):
    """A 32-bit hash function based on SHA1.

    Args:
        data (bytes): the data to generate 32-bit integer hash from.

    Returns:
        int: an integer hash value that can be encoded using 32 bits.
    """
    return struct.unpack('!I', hashlib.sha1(data.encode('utf8') if isinstance(data, str) else data).digest()[:4])[0]

def sha1_64bit(data):
    """A 64-bit hash function based on SHA1.

    Args:
        data (bytes): the data to generate 64-bit integer hash from.

    Returns:
        int: an integer hash value that can be encoded using 64 bits.
    """
    return struct.unpack('!Q', hashlib.sha1(data.encode('utf8') if isinstance(data, str) else data).digest()[:8])[0]

def random_uniform_32bit(data = None):
    """A 32-bit hash function which randomly generates a 32-bit integer
       regardless of the argument data. It is useful for debugging 
       purposes when having reproducible deterministic hash string 
       for a given key is not needed.

    Args:
        data (bytes): not used

    Returns:
        int: an integer hash value that can be encoded using 32 bits.
    """

    return random.randint(0, 2**32 - 1)

def random_uniform_64bit(data = None):
    """A 32-bit hash function which randomly generates a 32-bit integer
       regardless of the argument data. It is useful for debugging 
       purposes when having reproducible deterministic hash string 
       for a given key is not needed.

    Args:
        data (bytes): not used

    Returns:
        int: an integer hash value that can be encoded using 32 bits.
    """

    return random.randint(0, 2**64 - 1)


from builtins import object
import hashlib
import base64

class CompactHash(object):
    """compact sha can be used to calculate nomad gids"""
    def __init__(self, proto):
        self._proto = proto

    def b64digest(self):
        return base64.b64encode(self.digest(), b"-_")[:-2].decode("ascii")

    def b32digest(self):
        res = base64.b32encode(self.digest())
        return res[:res.index('=')].decode("ascii")

    def update(self, data):
        if type(data) == type(u""):
            data=data.encode("utf-8")
        return self._proto.update(data)

    def gid(self, prefix=""):
        """returns a nomad gid with the given prefix"""
        return prefix + self.b64digest()[:28]

    def __getattr__(self, name):
        return getattr(self._proto, name)

def sha224(*args, **kwargs):
    """CompactSha using sha224 for the checksums (non standard)"""
    return CompactHash(hashlib.sha224(*args,**kwargs))

def sha512(baseStr=None,*args,**kwargs):
    """Uses sha512 to calculate the gid (default in nomad)

    If you pass and argument it is immediately added to the checksum.
    Thus sha512("someString").gid("X") creates a gid in one go"""
    sha=CompactHash(hashlib.sha512(*args,**kwargs))
    if baseStr is not None:
        sha.update(baseStr)
    return sha

def md5(*args, **kwargs):
    """CompactSha using md5 for the checksums (non standard)"""
    return CompactHash(hashlib.md5(*args,**kwargs))

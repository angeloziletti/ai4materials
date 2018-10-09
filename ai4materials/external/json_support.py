"""
Module for Various functions to simplify and standardize dumping objects to json.


NOTE: this is taken from python-common in nomad-lab-base.
It is copied here to remove the dependency from nomad-lab-base.
For more info on python-common visit:
https://gitlab.mpcdf.mpg.de/nomad-lab/python-common

The author of this code is: Dr. Fawzi Roberto Mohamed
E-mail: mohamed@fhi-berlin.mpg.de

"""
from builtins import object
import json
from ai4materials.external.compact_sha import sha512
import numpy

def numpyEncoder(o):
    """new default function for json class so that numpy arrays and sets can be encoded"""
    # check if object is a numpy array
    if isinstance(o, numpy.ndarray):
        # ensure that we have an array with row-major memory order (C like)
        if not o.flags['C_CONTIGUOUS']:
            o = numpy.ascontiguousarray(o)
        return o.tolist()
        # see default method in python/json/encoder.py
    elif isinstance(o, set):
        return list(sorted(o))
    else:
        raise TypeError(repr(o) + " is not JSON serializable")

class ExtraIndenter(object):
    """Helper class to add extra indent at the beginning of every line"""
    def __init__(self, fStream, extraIndent):
        self.fStream = fStream
        self.indent = " " * extraIndent if extraIndent else  ""
    def write(self, val):
        i = 0
        while True:
            j = val.find("\n", i)
            if j == -1:
                self.fStream.write(val[i:])
                return
            j += 1
            self.fStream.write(val[i:j])
            self.fStream.write(self.indent)
            i = j

def jsonCompactF(obj, fOut, check_circular = False):
    """Dumps the object obj with a compact json representation using the utf_8 encoding 
    to the file stream fOut"""
    json.dump(obj, fOut, sort_keys = True, indent = None, separators = (',', ':'),
            ensure_ascii = False, check_circular = check_circular, default = numpyEncoder)

def jsonIndentF(obj, fOut, check_circular = False, extraIndent = None):
    """Dumps the object obj with an indented json representation using the utf_8 encoding 
    to the file stream fOut"""
    fStream = fOut
    if extraIndent:
        fStream = ExtraIndenter(fOut, extraIndent = extraIndent)
    json.dump(obj, fStream, sort_keys = True, indent = 2, separators = (',', ': '),
            ensure_ascii = False, check_circular = check_circular, default = numpyEncoder)

class DumpToStream(object):
    """transform a dump function in a stream"""
    def __init__(self, dumpF, extraIndent = None):
        self.baseDumpF = dumpF
        self.extraIndent = extraIndent
        self.indent = " " * extraIndent if extraIndent else  ""
        self.dumpF = self.dumpIndented if extraIndent else dumpF
    def dumpIndented(self, val):
        if type(val) == type(u""):
            val = val.encode("utf_8")
        i = 0
        while True:
            j = val.find("\n", i)
            if j == -1:
                self.baseDumpF(val[i:])
                return
            j += 1
            self.baseDumpF(val[i:j])
            self.baseDumpF(self.indent)
            i = j
    def write(self, val):
        self.dumpF(val)

def jsonCompactD(obj, dumpF, check_circular = False):
    """Dumps the object obj with a compact json representation using the utf_8 encoding 
    to the file stream fOut"""
    json.dump(obj, DumpToStream(dumpF), sort_keys = True, indent = None, separators = (', ', ': '),
            ensure_ascii = False, check_circular = check_circular, default = numpyEncoder)

def jsonIndentD(obj, dumpF, check_circular = False, extraIndent = None):
    """Dumps the object obj with an indented json representation using the utf_8 encoding 
    to the function dumpF"""
    json.dump(obj, DumpToStream(dumpF, extraIndent = extraIndent), sort_keys = True, indent = 2, separators = (',', ': '),
            ensure_ascii = False, check_circular = check_circular, encoding="utf_8", default = numpyEncoder)

def jsonCompactS(obj, check_circular = False):
    """returns a compact json representation of the object obj as a string"""
    return json.dumps(obj, sort_keys = True, indent = None, separators = (', ', ': '),
            ensure_ascii = False, check_circular = check_circular, encoding="utf_8", default = numpyEncoder)

def jsonIndentS(obj, check_circular = False, extraIndent = None):
    """retuns an indented json representation if the object obj as a string"""
    res = json.dumps(obj, sort_keys = True, indent = 2, separators = (',', ': '),
            ensure_ascii = False, check_circular = check_circular, encoding="utf_8", default = numpyEncoder)
    if extraIndent:
        indent = " " * extraIndent
        res = res.replace("\n", "\n" + indent)
    return res

def jsonDump(obj, path):
    """Dumps the object obj to an newly created utf_8 file at path"""
    kwds = dict()
    if sys.version_info.major > 2:
        kwds["encoding"] = "utf_8"
    with open(path, "w", **kwds) as f:
        jsonIndentF(obj, f)

class ShaStreamer(object):
    """a file like object that calculates one or more shas"""
    def __init__(self, shas = None):
        self.shas = shas
        if shas is None:
            self.shas = (sha512(),)
    def write(self, val):
        for sha in self.shas:
            sha.update(val)
    def b64digests(self):
        return [sha.b64digest() for sha in self.shas]

def addShasOfJson(obj, shas = None):
    """adds the jsonDump of obj to the shas"""
    streamer = ShaStreamer(shas)
    jsonCompactF(obj, streamer)
    return streamer

def normalizedJsonGid(obj, shas = None):
    """returns the gid of the standard formatted jsonDump of obj"""
    return ['j' + x for x in addShasOfJson(shas).b64digests()]

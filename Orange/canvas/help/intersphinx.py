"""
Parsers for intersphinx inventory files

Taken from `sphinx.ext.intersphinx`

"""

import re
import codecs
import zlib

b = str
UTF8StreamReader = codecs.lookup('utf-8')[2]


def read_inventory_v1(f, uri, join):
    f = UTF8StreamReader(f)
    invdata = {}
    line = next(f)
    projname = line.rstrip()[11:]
    line = next(f)
    version = line.rstrip()[11:]
    for line in f:
        name, type, location = line.rstrip().split(None, 2)
        location = join(uri, location)
        # version 1 did not add anchors to the location
        if type == 'mod':
            type = 'py:module'
            location += '#module-' + name
        else:
            type = 'py:' + type
            location += '#' + name
        invdata.setdefault(type, {})[name] = (projname, version, location, '-')
    return invdata


def read_inventory_v2(f, uri, join, bufsize=16*1024):
    invdata = {}
    line = f.readline()
    projname = line.rstrip()[11:].decode('utf-8')
    line = f.readline()
    version = line.rstrip()[11:].decode('utf-8')
    line = f.readline().decode('utf-8')
    if 'zlib' not in line:
        raise ValueError

    def read_chunks():
        decompressor = zlib.decompressobj()
        for chunk in iter(lambda: f.read(bufsize), b('')):
            yield decompressor.decompress(chunk)
        yield decompressor.flush()

    def split_lines(iter):
        buf = b('')
        for chunk in iter:
            buf += chunk
            lineend = buf.find(b('\n'))
            while lineend != -1:
                yield buf[:lineend].decode('utf-8')
                buf = buf[lineend+1:]
                lineend = buf.find(b('\n'))
        assert not buf

    for line in split_lines(read_chunks()):
        # be careful to handle names with embedded spaces correctly
        m = re.match(r'(?x)(.+?)\s+(\S*:\S*)\s+(\S+)\s+(\S+)\s+(.*)',
                     line.rstrip())
        if not m:
            continue
        name, type, prio, location, dispname = m.groups()
        if location.endswith('$'):
            location = location[:-1] + name
        location = join(uri, location)
        invdata.setdefault(type, {})[name] = (projname, version,
                                              location, dispname)
    return invdata

#!/usr/bin/env python

import sys
from sgmllib import SGMLParser
import optparse

def normalize(reference):
    return reference.replace("#", "\\#").replace("\n", " ").strip()

class ReferenceSet:
    def __init__(self):
        self.references = []
        self.keys = {}

    def addReference(self, key, reference):
        if self.keys.has_key(key):
            self.keys[key].append(reference)
        else:
            self.references.append([reference])
            self.keys[key] = self.references[-1]

    def printReferences(self, output):
        for refSet in self.references:
            output.write(normalize(refSet[0]))
            for r in refSet[1:]:
                output.write(" # %s" % normalize(r))
            output.write("\n")

class SGMLRefParser(SGMLParser):
    def __init__(self, referenceSet, verbose):
        SGMLParser.__init__(self)
        self.referenceSet = referenceSet
        self.inSegment = False
        self.verbose = verbose
        self.lastReference = None

    def start_doc(self, attributes):
        if self.verbose:
            sys.stderr.write("start_doc %s\n" % str(attributes))
        self.basekey = ""
        for a in attributes:
            if a[0] != "sysid":
                self.basekey += ("%s#" % a[1])

    def start_seg(self, attributes):
        if self.verbose:
            sys.stderr.write("start_seg %s\n" % str(attributes))
        for a in attributes:
            if a[0] == "id":
                self.segid = a[1]
        self.inSegment = True

    def end_seg(self):
        if self.verbose:
            sys.stderr.write("end_seg\n")
        if self.inSegment:
            self.referenceSet.addReference("%s#%s" % (self.basekey, self.segid), self.lastReference)
            self.inSegment = False
            self.lastReference = None

    def handle_data(self, data):
        if self.verbose:
            sys.stderr.write("data: %s\n" % data)
        if self.inSegment:
            if self.lastReference:
                self.lastReference += " %s" % (data.strip())
            else:
                self.lastReference = data.strip()
            

def main():
    usage = "usage: %prog [options] sgmfile"
    optionParser = optparse.OptionParser(usage=usage)
    optionParser.add_option("-o", "--out", metavar="file", dest="outFilename", help="output file")
    optionParser.add_option("-v", "--verbose", action="store_true", dest="verbose", default=False,
                            help="write every step to stderr, useful for finding errors in sgml files")
    (options, args) = optionParser.parse_args()

    if len(args) == 0:
        fpIn = sys.stdin
    else:
        try:
            fpIn = open(args[0])
        except IOError, ioError:
            optionParser.error("couldn't open %s, %s" % (args[0], ioError[1]))

    if not options.outFilename:
        fpOut = sys.stdout
    else:
        try:
            fpOut = open(options.outFilename, "w")
        except IOError, ioError:
            optionParser.error("couldn't open %s for writing, %s" % (options.outFilename, ioError[1]))
            
    referenceSet = ReferenceSet()
    parser = SGMLRefParser(referenceSet, options.verbose)
    parser.feed(fpIn.read())
    referenceSet.printReferences(fpOut)
    
if __name__ == "__main__":
    main()

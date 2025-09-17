#!/usr/bin/env python

import sys
from sgmllib import SGMLParser
import optparse
import re

debugPrint = sys.stderr.write
verbosePrint = sys.stderr.write

def normalize(reference):
    return reference.replace("#", "\\#").replace("\n", " ").strip()

class SGMLRefParser(SGMLParser):
    def __init__(self, sysid, targetLanguage, fpIn, fpOut, verbose):
        SGMLParser.__init__(self)
        self.sysid = sysid
        self.targetLanguage = targetLanguage
        self.fpIn = fpIn
        self.fpOut = fpOut
        self.verbose = verbose
        
    def __writeAttributes(self, attributes):
        for (attr, value) in attributes:
            self.fpOut.write(" %s=\"%s\"" % (attr, value))

    def __searchAttribute(self, attributes, attr):
        for (a, value) in attributes:
            if a == attr:
                return (a, value)
        return None

    def __setAttribute(self, attributes, attr, value):
        for i in xrange(len(attributes)):
            if attributes[i][0] == attr:
                attributes[i] = (attr, value)
                return
        attributes.append((attr, value))

    def start_refset(self, attributes):
        sys.stderr.write("Error: input must be a srcset\n")
        sys.exit(1)

    def start_srcset(self, attributes):
        self.fpOut.write("<tstset")
        self.__setAttribute(attributes, "trglang", self.targetLanguage)
        self.__writeAttributes(attributes)
        self.fpOut.write(">\n")

    def start_tstset(self, attributes):
        sys.stderr.write("Error: input must be a srcset\n")
        sys.exit(1)

    def start_doc(self, attributes):
        if self.verbose:
            verbosePrint("start_doc %s\n" % str(attributes))
        self.fpOut.write("<doc")
        self.__setAttribute(attributes, "sysid", self.sysid)
        self.__writeAttributes(attributes)
        self.fpOut.write(">\n")
        
    def start_hl(self, attributes):
        if self.verbose:
            verbosePrint("start_hl\n")
        self.fpOut.write("<hl>\n")
    
    def start_p(self, attributes):
        if self.verbose:
            verbosePrint("start_p\n")
        self.fpOut.write("<p>\n")

    def start_seg(self, attributes):
        if self.verbose:
            verbosePrint("start_seg %s\n" % str(attributes))

        self.fpOut.write("<seg")
        self.__writeAttributes(attributes)
        self.fpOut.write(">\n")

        self.fpOut.write(self.fpIn.next())

    def end_hl(self):
        if self.verbose:
            verbosePrint("end_hl\n")
        self.fpOut.write("</hl>\n")

    def end_p(self):
        if self.verbose:
            verbosePrint("end_p\n")
        self.fpOut.write("</p>\n")            

    def end_srcset(self):
        if self.verbose:
            verbosePrint("end_srcset\n")
        self.fpOut.write("</tstset>\n")

    def end_doc(self):
        if self.verbose:
            verbosePrint("end_doc\n")
        self.fpOut.write("</doc>\n")
        self.filterPassed = False

    def end_seg(self):
        if self.verbose:
            verbosePrint("end_seg\n")
        self.fpOut.write("</seg>\n")

def main():
    usage = "usage: %prog [options] [sgmfile]"
    optionParser = optparse.OptionParser(usage=usage)
    optionParser.add_option("-o", "--out", metavar="file", dest="outFilename", help="output file")
    optionParser.add_option("--id", metavar="sysid", dest="sysid", help="system id")
    optionParser.add_option("-s", "--source", metavar="sgmfile", dest="sourceFilename", help="Source file")
    optionParser.add_option("-l", "--targetLang", metavar="language", dest="targetLanguage", help="Set target language")
    optionParser.add_option("-v", "--verbose", action="store_true", dest="verbose", default=False,
                            help="write every step to stderr, useful for finding errors in sgml files")

    if len(sys.argv) == 1:
        optionParser.print_help()
        sys.exit(1)
    (options, args) = optionParser.parse_args()

    if len(args) == 0:
        fpIn = sys.stdin
    else:
        try:
            fpIn = open(args[0])
        except IOError, ioError:
            optionParser.error("couldn't open %s, %s" % (args[0], ioError[1]))

    if not options.sourceFilename:
        optionParser.error("a source filename should be given")
    try:
        fpSource = open(options.sourceFilename)
    except IOError, ioError:
        optionParser.error("couldn't open %s, %s" % (args[0], ioError[1]))

    if not options.targetLanguage:
        optionParser.error("a target language should be given")

    if not options.sysid:
        optionParser.error("a system id should be given")

    if not options.outFilename:
        fpOut = sys.stdout
    else:
        try:
            fpOut = open(options.outFilename, "w")
        except IOError, ioError:
            optionParser.error("couldn't open %s for writing, %s" % (options.outFilename, ioError[1]))

    parser = SGMLRefParser(options.sysid, options.targetLanguage, fpIn, fpOut, options.verbose)
    parser.feed(fpSource.read())
    
if __name__ == "__main__":
    main()

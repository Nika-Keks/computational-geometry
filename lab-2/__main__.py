import argparse
import sys
import numpy as np

from . import pset

def get_args(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("input", type=str, help="input file")
    parser.add_argument("-o", "--output", type=str, help="output file")

    return parser.parse_args(argv)

def main(argv):
    args = get_args(argv)
    
    # read input
    infile = open(args.input, "r")
    vlist = np.array([[int(x) for x in line.replace(" ", "").split(",")] for line in infile])
    infile.close()

    # build main support plane
    ps = pset.PointSet(vlist)
    res = ps.min_splane()

    # write result
    outfile = open(args.output, "w")
    outfile.write(str(res)[1:-1])
    outfile.close()

if __name__ == "__main__":
    main(sys.argv[1:])
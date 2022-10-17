import argparse
from ast import arg
import sys
import numpy as np

from . import obb

criterions = {
    "square": lambda a, b: a * b,
    "perimeter": lambda a, b: a + b
}

def main(args):

    # read input
    infile = open(args.inout, "r")
    v_list = np.array([[int(x) for x in line.replace(" ", "").split(",")] for line in infile])
    infile.close()

    # eval obb
    p = obb.Poligon(v_list, criterions[args.criterion])
    res = p.get_obb()

    # write results
    outfile = open(args.output, "w")
    outfile.write(str(res)[1:-1])
    outfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inout", type=str, help="input file")
    parser.add_argument("-o", "--output", type=str, default="out-obb", help="output file")
    parser.add_argument("-c", "--criterion", type=str, choices=criterions.keys(), default="perimeter", help=f"criterion type {criterions.keys()}")
    
    main(parser.parse_args(sys.argv[1:]))


import sys

from r0714272 import r0714272

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Please provide the name of the csv file that describes a distance matrix")
        exit(1)

    filename = "{}.csv".format(sys.argv[1])
    r0714272().optimize(filename)
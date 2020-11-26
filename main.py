import sys
import matplotlib.pyplot as plt

from r0714272 import r0714272

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Please provide the name of the csv file that describes a distance matrix")
        exit(1)

    filename = "{}.csv".format(sys.argv[1])
    r0714272().optimize(filename)

    csv_content = [line.split(',') for line in open("r0714272.csv").read().split("\n")[2:] if line.strip() != ""]
    iterations = [int(elems[0]) for elems in csv_content]
    # elapsed_time = [float(elems[1]) for elems in csv_content]
    mean_values = [float(elems[2]) for elems in csv_content]
    best_values = [float(elems[3]) for elems in csv_content]

    print("Last best value: {}".format(best_values[-1]))
    print("Actual best value: {}".format(min(best_values)))

    plt.plot(iterations, mean_values, label="Mean value")
    plt.plot(iterations, best_values, label="Best value")
    plt.xticks(iterations)
    plt.legend()

    plt.show()


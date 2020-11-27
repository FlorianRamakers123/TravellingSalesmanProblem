import sys
import matplotlib.pyplot as plt

from r0714272 import r0714272

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Please provide the name of the csv file that describes a distance matrix")
        exit(1)

    filename = "{}.csv".format(sys.argv[1])
    r = r0714272()
    try:
        r.optimize(filename)
    except KeyboardInterrupt:
        pass

    csv_content = [line.split(',') for line in open("r0714272.csv").read().split("\n")[2:] if line.strip() != ""]
    iterations = [int(elems[0]) for elems in csv_content]
    # elapsed_time = [float(elems[1]) for elems in csv_content]
    mean_values = [float(elems[2]) for elems in csv_content]
    best_values = [float(elems[3]) for elems in csv_content]

    last_best = best_values[-1]
    actual_best = min(best_values)
    print("Last best value: {}".format(last_best))
    print("Actual best value: {}".format(actual_best))

    f = open("runs.txt", 'a')
    f.write("-------------\n{}\nbest={}\nactual best={}\nparameters:{}\n".format(filename, last_best, actual_best, r.ap.__dict__))

    plt.plot(iterations, mean_values, label="Mean value")
    plt.plot(iterations, best_values, label="Best value")
    plt.xticks(iterations[0:len(iterations):max(1,int(len(iterations) / 20))])
    plt.legend()

    plt.show()


import numpy as np
import matplotlib.pyplot as plt


def plot_permutation(perm):
    coords_s = open("Coord194.csv").readlines()
    coords = [(int(line.split(",")[1].strip()), int(line.split(",")[2].strip())) for line in coords_s]

    for i in range(perm.shape[0]):

        c1 = coords[perm[i]]
        c2 = coords[perm[(i+1) % perm.shape[0]]]

        plt.plot([c1[0], c2[0]], [c1[1], c2[1]], marker='o', c='r')
        if i == 0:
            plt.scatter(c1[0], c1[1], s=100, c='b')
        if i == 1:
            plt.scatter(c1[0], c1[1], s=100, c='g')
        #plt.scatter(c1[0], c1[1], s=100, c='r')


    plt.show()




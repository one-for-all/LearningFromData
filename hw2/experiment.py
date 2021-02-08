import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = np.load("data.npy")
    assert(len(data[0]) == 100000)

    # plt.hist(data[2])
    # plt.show()
    print(data[2][:1000])
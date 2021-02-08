# Run the Hoeffding Inequality experiment
# for problem 1 and 2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


if __name__ == "__main__":
    v_1 = []
    v_rand = []
    v_min = []
    N_trials = 1000
    N_samples = 1000
    N_flips = 10
    for _ in tqdm(range(N_trials)):
        heads = [np.sum(np.random.rand(N_flips) > 0.5) for _ in range(N_samples)]
        v_1.append(heads[0]/N_flips)
        v_rand.append(np.random.choice(heads)/N_flips)
        v_min.append(np.min(heads)/N_flips)

    assert(len(v_1) == N_trials)
    print("average value of v_min: {}".format(np.mean(v_min)))

    with open("data2.npy", 'wb') as f:
        np.save(f, [v_1, v_rand, v_min])

    # Test Hoeffding Inequality\
    epsilons = np.arange(0, 0.6, 0.1)
    probs = 2*np.e**(-2*epsilons**2*N_flips)

    mu = 0.5  # fair coin probability of head
    count_1 = [0 for _ in range(len(epsilons))]
    count_rand = [0 for _ in range(len(epsilons))]
    count_min = [0 for _ in range(len(epsilons))]
    for v_1_item, v_rand_item, v_min_item in zip(v_1, v_rand, v_min):
        for idx, epsilon in enumerate(epsilons):
            if abs(v_1_item - mu) > epsilon:
                count_1[idx] += 1
            if abs(v_rand_item - mu) > epsilon:
                count_rand[idx] += 1
            if abs(v_min_item - mu) > epsilon:
                count_min[idx] += 1

    prob_1 = np.array(count_1)/N_trials
    prob_rand = np.array(count_rand)/N_trials
    prob_min = np.array(count_min)/N_trials

    # Plot
    plt.plot(epsilons, probs, label="target")
    plt.plot(epsilons, prob_1, label="c_1")
    plt.plot(epsilons, prob_rand, label="c_rand")
    plt.plot(epsilons, prob_min, label="c_min")
    plt.legend()
    plt.show()

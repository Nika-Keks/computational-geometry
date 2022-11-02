import numpy as np

from matplotlib import pyplot as plt

def ss_nss(vlist: np.array, ids: np.array):
    v_len = len(vlist)
    support_side = vlist[ids[0] % v_len] - vlist[(ids[0]-1)%v_len]
    ss_norm = support_side.dot(support_side)
    support_side = support_side.astype(np.float64) / np.sqrt(np.float64(ss_norm))
    norm_support_side = np.array([support_side[1], -support_side[0]])

    return support_side, norm_support_side

def get_obb(vlist: np.array, ids: np.array):
    ss, nss = ss_nss(vlist, ids)

    xs = []
    for v1, v2 in zip(vlist[ids], vlist[ids[range(-1, 3)]]):
        A = np.array([nss, ss])
        b = np.array([nss.dot(v1), ss.dot(v2)])
        x = np.linalg.inv(A).dot(b)
        xs.append(x)
        nss, ss = ss, nss
    xs = np.array(xs)[range(-1, 4)]

    return xs


def plot_obb(vlist: np.array, ids: np.array, title=""):
    plt.scatter(vlist[:, 0], vlist[:, 1])
    for i, v in enumerate(vlist):
        plt.annotate(f"v{i}", v)
    
    
    xs = get_obb(vlist, ids)

    plt.plot(xs[:, 0], xs[:, 1])
    plt.title(title)
    plt.grid()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll

def plot_shap_heatmap(x, y, shap, cmap='plasma'):
    norm = plt.Normalize(shap.min(), shap.max())
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = mcoll.LineCollection(segments, array=shap, norm=norm,
                              linewidth=3, alpha=1.0, cmap=cmap)

    ax = plt.gca()
    ax.add_collection(lc)

    plt.colorbar(lc)
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min() * 1.1, y.max() * 1.1)
    plt.show()



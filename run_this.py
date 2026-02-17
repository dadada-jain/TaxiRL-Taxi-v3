import os
import matplotlib.pyplot as plt
import numpy as np
from DQN import DQN
from Qlearning import Q_learning


def moving_average(arr, window):
    """Moving average with a fixed window size."""
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) < window:
        return arr
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(arr, kernel, mode="valid")  # keep only full windows


if __name__ == "__main__":
    total_rewards1, illegals1 = Q_learning()
    total_rewards2, illegals2 = DQN()

    window = 200

    out_dir = "assets"
    os.makedirs(out_dir, exist_ok=True)

    max_ep = max(len(total_rewards1), len(total_rewards2))
    step = max(200, max_ep // 5)  # ~5 ticks

    # ===== Plot 1: Episode return =====
    ret_ma1 = moving_average(total_rewards1, window)
    ret_ma2 = moving_average(total_rewards2, window)

    x1 = range(window - 1, window - 1 + len(ret_ma1))
    x2 = range(window - 1, window - 1 + len(ret_ma2))

    plt.figure(figsize=(12, 5))
    plt.plot(x1, ret_ma1, linewidth=2.2, label=f"Q-learning MA({window})")
    plt.plot(x2, ret_ma2, linewidth=2.2, label=f"DQN MA({window})")

    ax = plt.gca()
    ax.set_xticks(list(range(0, max_ep + 1, step)))
    ax.set_yticks([-1000, -800, -600, -400, -200, 0, 10])

    plt.grid(alpha=0.3)
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Episode return")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "compare_return.png"), dpi=220)
    plt.show()

    # ===== Plot 2: Illegal actions =====
    ill_ma1 = moving_average(illegals1, window)
    ill_ma2 = moving_average(illegals2, window)

    xi1 = range(window - 1, window - 1 + len(ill_ma1))
    xi2 = range(window - 1, window - 1 + len(ill_ma2))

    plt.figure(figsize=(12, 5))
    plt.plot(xi1, ill_ma1, linewidth=2.2, label=f"Q-learning illegal MA({window})")
    plt.plot(xi2, ill_ma2, linewidth=2.2, label=f"DQN illegal MA({window})")

    ax = plt.gca()
    ax.set_xticks(list(range(0, max_ep + 1, step)))

    plt.grid(alpha=0.3)
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Illegal actions per episode")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "compare_illegal.png"), dpi=220)
    plt.show()

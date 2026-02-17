import gymnasium as gym
import numpy as np
import imageio.v2 as imageio
def Q_learning():
    alpha = 0.1 # 学习率：控制参数更新步长
    gamma = 0.99  # 折扣因子：越接近1越重视长期回报
    epsilon = 1.0  # ε-greedy：探索概率（训练初期高探索）
    epsilon_min = 0.001
    epsilon_decay = 0.999  # 每回合衰减：逐步从“探索”过渡到“利用”
    N_EPISODES = 5000  # 训练回合数
    n_states = 500  # Taxi-v3 离散状态数（0~499）
    n_actions = 6 # 有上下左右、让乘客上下车 6 种动作
    Q = np.zeros((n_states, n_actions)) # 初始化 0

    env = gym.make("Taxi-v3")

    total_rewards = []  # 每回合累计回报（评估学习效果）
    illegals = []  # 每回合非法动作次数（Taxi-v3 中 reward=-10 常对应非法 pickup/dropoff）
    epsilons = []  # 记录 ε 衰减轨迹，便于分析探索程度

    for episode in range(N_EPISODES):
        observation, info = env.reset()
        episode_over = False
        step_count, illegal, total_reward = 0, 0, 0  # 步数 非法次数 分数
        while not episode_over:
            state = observation
            # ε-greedy：以 ε 概率随机探索，否则选择当前 Q 最大的动作
            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[state]))

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 若回合结束，则下一状态价值为 0（无未来回报）
            best_next = 0.0 if done else np.max(Q[next_obs])

            # TD 更新：Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]
            Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])

            observation = next_obs
            episode_over = done

            step_count = step_count + 1
            if reward == -10:
                illegal = illegal + 1
            total_reward = total_reward + reward

        total_rewards.append(total_reward)
        illegals.append(illegal)
        epsilons.append(epsilon)
        # 将新一局数据存入
        epsilon = max(epsilon_min, epsilon * epsilon_decay) # 更新随机概率

    env.close()
    # === policy demo：用训练好的模型纯贪心跑一局，并导出 gif ===
    demo_env = gym.make("Taxi-v3", render_mode="rgb_array")
    frames = []

    observation, info = demo_env.reset()
    frames.append(demo_env.render())
    done = False
    steps = 0
    while not done :
        action = int(np.argmax(Q[observation]))
        observation, reward, terminated, truncated, info = demo_env.step(action)
        done = terminated or truncated
        frames.append(demo_env.render())
        steps = steps + 1
    demo_env.close()
    imageio.mimsave("demo_Q-learning.gif", frames, fps=6)
    # ===== GIF 导出结束 =====

    return  total_rewards, illegals
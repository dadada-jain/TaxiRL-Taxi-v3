import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import imageio.v2 as imageio

def DQN():
    learn_rate = 1e-3  # 学习率：控制参数更新步长（过大可能不稳定，过小收敛慢）
    gamma = 0.99  # 折扣因子：越接近1越重视长期回报
    epsilon = 1.0  # ε-greedy：探索概率（训练初期高探索）
    epsilon_min = 0.001
    epsilon_decay = 0.999  # 每回合衰减：逐步从“探索”过渡到“利用”
    N_EPISODES = 5000  # 训练回合数
    n_states = 500  # Taxi-v3 离散状态数（0~499）
    i=0  # 用于 target network 的同步计数

    env = gym.make("Taxi-v3")

    total_rewards = []  # 每回合累计回报（评估学习效果）
    illegals = []  # 每回合非法动作次数（Taxi-v3 中 reward=-10 常对应非法 pickup/dropoff）
    epsilons = []  # 记录 ε 衰减轨迹，便于分析探索程度

    def obs_to_onehot(obs: int, n_states: int):
        x = np.zeros(n_states, dtype=np.float32)
        x[obs] = 1.0
        return x

    class MyModel(nn.Module):
        def __init__(self, in_dim = 500, hidden_dim = 128, out_dim = 6):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
            )
        def forward(self, x):
            return self.net(x)
    # 构建网络 最小MLP baseline

    memory1 = deque(maxlen=50000)  # 经验回放池：打破序列相关性，提升训练稳定性
    model = MyModel()  # online network：用于选动作 & 学习 Q(s,a)
    target_net = MyModel()  # target network：用于计算 TD target（减少目标漂移）
    target_net.load_state_dict(model.state_dict())
    target_net.eval()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    loss_fn = nn.MSELoss()
    batch_size = 64
    start_learning = 1000  # 先积累一定经验再开始更新，避免早期样本过少导致不稳定

    for episode in range(N_EPISODES):
        observation, info = env.reset()
        episode_over = False
        step_count, illegal, total_reward = 0, 0, 0

        while not episode_over:
            x = torch.from_numpy(obs_to_onehot(observation, 500)).unsqueeze(0).float()  # (1,500)

            # ε-greedy：以 ε 概率随机探索，否则选择当前 Q 最大的动作
            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                q_values = model(x)  # (1,6)
                action = torch.argmax(q_values, dim=1).item()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 存一条 transition: (s, a, r, s', done)
            memory1.append((observation, action, reward, next_obs, done))

            observation = next_obs
            episode_over = done

            # 训练更新：经验足够后才开始学
            if len(memory1) >= start_learning:
                i=i+1
                # 从回放池采样一个 mini-batch（近似 i.i.d.）
                batch = random.sample(memory1, batch_size)
                s1, a_batch, r_batch, s2, d_batch = zip(*batch)

                s1 = torch.from_numpy(np.stack([obs_to_onehot(si, n_states) for si in s1])).float()  # (B,500)
                s2 = torch.from_numpy(np.stack([obs_to_onehot(si, n_states) for si in s2])).float()  # (B,500)

                a_batch = torch.tensor(a_batch).long().unsqueeze(1)  # (B,1)
                r_batch = torch.tensor(r_batch).float().unsqueeze(1)  # (B,1)
                d_batch = torch.tensor(d_batch).float().unsqueeze(1)  # (B,1) done: True->1

                # 当前网络给出的 Q(s,a)：用 gather 取出每条样本实际执行动作 a 的那一列
                q_pred = model(s1).gather(1, a_batch)  # (B,1)

                # TD target：r + γ * (1-done) * max_a' Q_target(s', a')
                with torch.no_grad(): # 计算 TD target 不需要梯度（避免梯度流入 target_net）
                    q_next = target_net(s2).max(dim=1, keepdim=True)[0]  # (B,1)
                    target = r_batch + gamma * (1 - d_batch) * q_next

                loss = loss_fn(q_pred, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 周期性同步 target 网络（硬更新）：降低训练振荡
                if i % 100 == 0:
                    target_net.load_state_dict(model.state_dict())

            step_count = step_count + 1
            if reward == -10:
                illegal = illegal + 1

            total_reward = total_reward + reward

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        total_rewards.append(total_reward)
        illegals.append(illegal)
        epsilons.append(epsilon)

    env.close()
    # === policy demo：用训练好的模型纯贪心跑一局，并导出 gif ===
    demo_env = gym.make("Taxi-v3", render_mode="rgb_array")
    frames = []

    obs, info = demo_env.reset()
    frames.append(demo_env.render())

    done = False
    steps = 0

    with torch.no_grad(): # 推理阶段不需要梯度（更快、更省内存）
        while not done:
            x = torch.from_numpy(obs_to_onehot(obs, 500)).unsqueeze(0).float()
            action = int(torch.argmax(model(x), dim=1).item())  # 纯贪心

            obs, reward, terminated, truncated, info = demo_env.step(action)
            done = terminated or truncated

            frames.append(demo_env.render())
            steps = steps + 1

    demo_env.close()
    imageio.mimsave("demo_DQN.gif", frames, fps=6)
    # ===== GIF 导出结束 =====
    return  total_rewards, illegals
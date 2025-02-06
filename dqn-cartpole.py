import gymnasium as gym
from gymnasium.wrappers import RecordVideo  # 新增导入
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import swanlab

# 设置随机数种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# 定义Q网络
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.to(device)  # 将网络移到指定设备
    
    def forward(self, x):
        return self.fc(x)

# DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.q_net = QNetwork(state_dim, action_dim)       # 当前网络
        self.target_net = QNetwork(state_dim, action_dim)  # 目标网络
        self.best_net = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.replay_buffer = deque(maxlen=10000)           # 经验回放缓冲区
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 0.1
        self.update_target_freq = 100  # 目标网络更新频率
        self.step_count = 0
        self.log_loss = 0
        self.best_reward = 0

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 2)  # CartPole有2个动作（左/右）
        else:
            state_tensor = torch.FloatTensor(state).to(device)
            q_values = self.q_net(state_tensor)
            return q_values.cpu().detach().numpy().argmax()

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # 从缓冲区随机采样
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # 计算当前Q值
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # 计算目标Q值（使用目标网络）
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # 计算损失并更新网络
        loss = nn.MSELoss()(current_q, target_q)
        # self.log_loss += loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 定期更新目标网络
        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            # 使用深拷贝更新目标网络参数
            self.target_net.load_state_dict({
                k: v.clone() for k, v in self.q_net.state_dict().items()
            })

    def save_model(self, path="best_model.pth"):
        torch.save(self.q_net.state_dict(), path)
        print(f"Model saved to {path}")
        
# 训练过程
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQNAgent(state_dim, action_dim)


# 初始化SwanLab日志记录器
swanlab.init(
    project="RL-All-In-One",
    experiment_name="DQN-CartPole-v1",
    config={
        "state_dim": state_dim,
        "action_dim": action_dim,
        "batch_size": agent.batch_size,
        "gamma": agent.gamma,
        "epsilon": agent.epsilon,
        "update_target_freq": agent.update_target_freq,
        "replay_buffer_size": agent.replay_buffer.maxlen,
        "learning_rate": agent.optimizer.param_groups[0]['lr'],
        "device": device,
        "episode": 300,
    },
)

for episode in range(swanlab.config["episode"]):
    state = env.reset()[0] # 获取初始状态
    total_reward = 0

    # 开启循环
    while True:
        action = agent.choose_action(state)  # 选择动作（有概率随机探索）
        next_state, reward, done, _, _ = env.step(action)  # 执行动作，获取下一个状态、奖励、是否结束等信息
        agent.store_experience(state, action, reward, next_state, done)  # 存储经验
        origin_q_net = agent.q_net.state_dict().copy()  # 创建当前网络的副本
        agent.train()  # 训练网络，更新参数
        total_reward += reward  # 累加奖励
        state = next_state  # 更新状态
        if done:  # 如果结束，则跳出循环
            if total_reward > agent.best_reward:  # 如果当前奖励大于最佳奖励
                agent.best_reward = total_reward  # 更新最佳奖励
                # 深拷贝当前最优模型的参数
                agent.best_net.load_state_dict({
                    k: v.clone() for k, v in origin_q_net.items()
                })
                agent.save_model()  # 保存最佳模型
                print("save model, best reward: ", agent.best_reward)
            break

    
    print(f"Episode: {episode}, Reward: {total_reward}, Best Reward: {agent.best_reward}")
    
    swanlab.log(
        {
            "train/reward": total_reward,
            "train/best_reward": agent.best_reward
        },
        step=episode,
    )

# 测试并录制视频
agent.epsilon = 0  # 关闭探索策略
test_env = gym.make('CartPole-v1', render_mode='rgb_array')
test_env = RecordVideo(test_env, "./dqn_videos", episode_trigger=lambda x: True)  # 保存所有测试回合
agent.q_net.load_state_dict(agent.best_net.state_dict())  # 使用最佳模型

for episode in range(3):  # 录制3个测试回合
    state = test_env.reset()[0]
    total_reward = 0
    steps = 0
    
    while True:
        action = agent.choose_action(state)
        next_state, reward, done, _, _ = test_env.step(action)
        total_reward += reward
        state = next_state
        steps += 1
        
        # 限制每个episode最多1500步,约30秒,防止录制时间过长
        if done or steps >= 1500:
            break
    
    print(f"Test Episode: {episode}, Reward: {total_reward}")

test_env.close()
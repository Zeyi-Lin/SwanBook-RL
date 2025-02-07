import gymnasium as gym
from gymnasium.wrappers import RecordVideo, ResizeObservation, ClipReward, GrayscaleObservation, FrameStackObservation
import ale_py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import swanlab
import os

# 设置随机数种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

gym.register_envs(ale_py)

# 设置设备
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"Using device: {device}")

# 修改Q网络为CNN结构
class QNetwork(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
    
    def forward(self, x):
        x = x / 255.0  # 归一化
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# DQN Agent
class DQNAgent:
    def __init__(self, action_dim):
        self.q_net = QNetwork(action_dim).to(device)
        self.target_net = QNetwork(action_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.best_net = QNetwork(action_dim).to(device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=0.00025)
        self.replay_buffer = deque(maxlen=100000)  # 增大回放缓冲区
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.update_target_freq = 1000  # 增加目标网络更新间隔
        self.step_count = 0
        self.best_reward = 0
        self.best_avg_reward = 0
        self.eval_episodes = 3

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.q_net.fc[-1].out_features)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.q_net(state_tensor)
            return q_values.cpu().detach().numpy().argmax()

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self, episode):
        if len(self.replay_buffer) < self.batch_size:
            return
        
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
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # print(f"Episode {episode}, Step {self.step_count}, Loss: {loss.item()}")

        # 定期更新目标网络
        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            # 使用深拷贝更新目标网络参数
            self.target_net.load_state_dict({
                k: v.clone() for k, v in self.q_net.state_dict().items()
            })
            
        return loss.item()

    def save_model(self, path="./output/best_model.pth"):
        if not os.path.exists("./output"):
            os.makedirs("./output")
        torch.save(self.q_net.state_dict(), path)
        print(f"Model saved to {path}")
        
    def evaluate(self, env):
        """评估当前模型的性能"""
        original_epsilon = self.epsilon
        self.epsilon = 0  # 关闭探索
        total_rewards = []
        max_steps = 1500  # 添加最大步数限制

        for _ in range(self.eval_episodes):
            # 使用预处理后的环境
            eval_env = make_env('ALE/Breakout-v5')  # 使用相同的预处理函数
            state = eval_env.reset()[0]
            episode_reward = 0
            steps = 0  # 记录步数
            while True:
                action = self.choose_action(state)
                next_state, reward, done, _, _ = eval_env.step(action)
                episode_reward += reward
                state = next_state
                steps += 1
                if done or steps >= max_steps:  # 添加步数限制条件
                    break
            eval_env.close()  # 关闭环境
            total_rewards.append(episode_reward)

        self.epsilon = original_epsilon  # 恢复探索
        return np.mean(total_rewards)

# 修改训练过程
def make_env(env_name, seed=None):
    env = gym.make(env_name)
    if seed is not None:
        env.seed(seed)
    env = ResizeObservation(env, (84, 84))
    env = GrayscaleObservation(env)
    env = FrameStackObservation(env, 4)
    env = ClipReward(env, min_reward=-1.0, max_reward=1.0)
    return env

# 初始化环境和智能体
env = make_env('ALE/Breakout-v5')
action_dim = env.action_space.n
agent = DQNAgent(action_dim)

# 加载预训练模型
model_path = "./output/best_model.pth"
agent.q_net.load_state_dict(torch.load(model_path))
print(f"Loaded model from {model_path}")

# 测试并录制视频
agent.epsilon = 0  # 关闭探索策略
test_env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')
# 添加与训练时相同的预处理包装器
test_env = ResizeObservation(test_env, (84, 84))
test_env = GrayscaleObservation(test_env)
test_env = FrameStackObservation(test_env, 4)
test_env = ClipReward(test_env, min_reward=-1.0, max_reward=1.0)
test_env = RecordVideo(test_env, "./dqn_videos", episode_trigger=lambda x: True)

for episode in range(3):  # 录制3个测试回合
    state = test_env.reset()[0]
    total_reward = 0
    steps = 0
    
    while True:
        # 确保状态张量的形状正确
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        action = agent.choose_action(state)
        next_state, reward, done, _, _ = test_env.step(action)
        total_reward += reward
        state = next_state
        steps += 1
        
        if done or steps>=10000:  # 限制每个episode最多2000步
            break
    
    print(f"Test Episode: {episode}, Reward: {total_reward}, Steps: {steps}")

test_env.close()
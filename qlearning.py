import numpy as np

# 定义环境参数
GRID_SIZE = 4  # 网格大小
ACTIONS = ['up', 'down', 'left', 'right']  # 动作列表
NUM_ACTIONS = 4  # 动作数量
GOAL = (3, 3)  # 目标状态

# Q-Learning参数
alpha = 0.1  # 学习率
gamma = 0.9  # 折扣因子
epsilon = 0.1  # 探索概率
episodes = 1000  # 训练轮数

# 初始化Q表（状态为网格坐标，每个状态有4个动作）
Q = np.zeros((GRID_SIZE, GRID_SIZE, NUM_ACTIONS))

# 定义动作到坐标变化的映射
action_effects = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}

# ε-greedy策略选择动作
def choose_action(state):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(NUM_ACTIONS)  # 随机探索
    else:
        return np.argmax(Q[state[0], state[1], :])  # 选择最优动作

# Q-Learning训练过程
for episode in range(episodes):
    state = (0, 0)  # 初始状态
    while state != GOAL:
        action = choose_action(state)
        dx, dy = action_effects[ACTIONS[action]]
        next_state = (state[0] + dx, state[1] + dy)
        
        # 检查是否越界或碰到障碍（这里简化处理）
        if next_state[0] < 0 or next_state[0] >= GRID_SIZE or next_state[1] < 0 or next_state[1] >= GRID_SIZE:
            reward = -1
            next_state = state  # 保持原状态
        else:
            reward = 10 if next_state == GOAL else 0
        
        # 更新Q值
        old_q = Q[state[0], state[1], action]
        max_next_q = np.max(Q[next_state[0], next_state[1], :])
        new_q = old_q + alpha * (reward + gamma * max_next_q - old_q)
        Q[state[0], state[1], action] = new_q
        
        state = next_state  # 转移到下一状态

# 测试训练后的策略
state = (0, 0)
path = [state]
while state != GOAL:
    action = np.argmax(Q[state[0], state[1], :])  # 直接选择最优动作
    dx, dy = action_effects[ACTIONS[action]]
    next_state = (state[0] + dx, state[1] + dy)
    path.append(next_state)
    state = next_state

print("最优路径:", path)
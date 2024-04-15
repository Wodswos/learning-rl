import time

import pygame
import random
import numpy as np
import gymnasium as gym


class SnakeEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # 初始化Pygame
        pygame.init()
        # 屏幕宽高
        self.SCREEN_WIDTH = 240
        self.SCREEN_HEIGHT = 240
        # 蛇的方块大小
        self.snakeCell = 10
        # 创建窗口
        self.screen = pygame.display.set_mode(
            (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        )
        pygame.display.set_caption("Snake_Game")
        self.action_space = gym.spaces.Discrete(4)  # 动作空间为4
        self.observation_space = gym.spaces.Box(
            low=0, high=7, shape=(self.SCREEN_WIDTH, self.SCREEN_HEIGHT),
            dtype=np.uint8
        )

    # 重启
    def reset(self):
        """
        重置蛇和食物的位置
        """
        # 蛇的初始位置
        self.snake_head = [100, 50]
        self.snake_body = [
            [100, 50],
            [100 - self.snakeCell, 50],
            [100 - self.snakeCell * 2, 50],
        ]
        self.len = 3

        # 食物的初始位置
        self.food_pos = [
            random.randint(1, self.SCREEN_WIDTH // 10 - 1) * 10,
            random.randint(1, self.SCREEN_HEIGHT // 10 - 1) * 10,
        ]

        return self._get_observation()

    # 根据当前状态 和action 执行动作
    def step(self, action):
        # 定义动作到方向的映射
        directionDict = {
            "LEFT": [1, 0],
            "RIGHT": [-1, 0],
            "UP": [0, -1],
            "DOWN": [0, 1],
        }

        action_to_direction = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        directionTarget = action_to_direction[action]
        nextPosDelay = (
            np.array(directionDict[directionTarget]) * self.snakeCell
        )  # 加的位置
        self.snake_head = list(np.array(self.snake_body[0]) + nextPosDelay)

        if self.snake_head in self.snake_body:
            return self._get_observation(), 0, True, False, {}
        self.snake_body.insert(0, self.snake_head)

        # 如果是吃到食物，就重新刷新果子，同时长度 +1
        if self.food_pos == self.snake_head:
            self.food_pos = [
                random.randrange(1, (self.SCREEN_WIDTH // 10)) * 10,
                random.randrange(1, (self.SCREEN_HEIGHT // 10)) * 10,
            ]
            self.len += 1

        # 弹出
        while self.len < len(self.snake_body):
            self.snake_body.pop()
        # 奖励
        reward, done = self._get_reward()
        truncated = True

        return self._get_observation(), reward, truncated, done, {}

    # 渲染
    def render(self, mode="human"):
        # 实现可视化
        screen = self.screen
        # 颜色定义
        WHITE = (255, 255, 255)
        GREEN = (0, 255, 0)
        RED = (255, 0, 0)

        # 清空屏幕
        screen.fill(WHITE)

        # 画蛇和食物
        for pos in self.snake_body:
            pygame.draw.rect(
                screen,
                GREEN,
                pygame.Rect(pos[0], pos[1], self.snakeCell, self.snakeCell),
            )
        pygame.draw.rect(
            screen,
            RED,
            pygame.Rect(
                self.food_pos[0], self.food_pos[1],
                self.snakeCell, self.snakeCell
            ),
        )

        pygame.display.update()

    # 获取奖励
    def _get_reward(self):

        # 计算奖励
        reward = 0
        done = False

        # 检查蛇是否吃到食物
        if self.snake_head:
            reward += 10

        # 检查蛇是否撞到墙壁或自身
        head = self.snake_head
        if (
            head[0] < 0
            or head[0] > self.SCREEN_WIDTH - 10
            or head[1] < 0
            or head[1] > self.SCREEN_HEIGHT - 10
        ):
            reward = -10
            done = True

        return reward, done

    # 获取当前观察空间
    def _get_observation(self):
        # 获取窗口内容作为观察值
        observation = pygame.display.get_surface()
        # 将观察值调整为指定的宽度和高度
        # observation = pygame.transform.scale(
        #     observation, (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        # )
        return observation


def main():
    snake_env = SnakeEnv()
    snake_env.reset()
    done = False
    while not done:
        # 获取事件
        for event in pygame.event.get():
            # 处理退出事件
            if event.type == pygame.QUIT:
                pygame.quit()
                done = True
        # 从动作空间随机获取一个动作
        action = snake_env.action_space.sample()
        screen, reward, truncated, done, _ = snake_env.step(action)
        snake_env.render()
        time.sleep(0.03)


if __name__ == "__main__":
    main()

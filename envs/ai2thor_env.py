import gym
import numpy as np
from gym import spaces
from thortils import launch_controller

class AI2ThorEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        # 初始化 AI2-THOR 控制器
        self.controller = launch_controller({"scene": config.scene, "VISIBILITY_DISTANCE": 3.0, "headless": False})

        # self.controller = ai2thor.controller.Controller(
        #     width=config.size[0],
        #     height=config.size[1],
        #     scene=config.scene,  # 例如 "FloorPlan1"
        #     renderInstanceSegmentation=False,
        #     renderDepthImage=False,
        #     renderClassImage=False,
        #     agentMode="default"
        # )
        self.action_list = config.action_list  # 例如 ["MoveAhead", "RotateLeft", ...]
        self.action_space = spaces.Discrete(len(self.action_list))
        # 假设观测为 RG
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(config.size[1], config.size[0], 3), dtype=np.uint8
        )
        self.max_steps = getattr(config, "time_limit", 200)
        self.current_step = 0

    def reset(self):
        self.controller.reset(self.controller.scene)
        self.current_step = 0
        event = self.controller.step(action="Pass")
        obs = self._get_obs(event)
        return obs

    def step(self, action):
        action_name = self.action_list[action]
        event = self.controller.step(action=action_name)
        obs = self._get_obs(event)
        reward, done = self._compute_reward_done(event)
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
        info = {}
        return obs, reward, done, info

    def _get_obs(self, event):
        # 返回 obs 字典，Dreamer3 兼容
        obs = {
            "image": event.frame  # shape: (H, W, 3), dtype: uint8
            # 你可以添加更多观测，如深度、目标信息等
        }
        # Dreamer3 需要 is_first/is_terminal 字段
        obs["is_first"] = np.array([self.current_step == 0], dtype=bool)
        obs["is_terminal"] = np.array([False], dtype=bool)  # 终止时在 step 里处理
        return obs

    def _compute_reward_done(self, event):
        # 你需要根据任务目标自定义 reward 和 done
        # 这里只是一个示例，实际请根据你的任务逻辑实现
        reward = 0.0
        done = False
        # 例如：到达目标物体
        # if event.metadata["lastActionSuccess"] and ...:
        #     reward = 1.0
        #     done = True
        return reward, done

    @property
    def action_space(self):
        return self._action_space

    @action_space.setter
    def action_space(self, value):
        self._action_space = value

    @property
    def observation_space(self):
        return self._observation_space

    @observation_space.setter
    def observation_space(self, value):
        self._observation_space = value

if __name__ == "__main__":
    import argparse

    class DummyConfig:
        # 你可以根据实际情况修改这些参数
        size = (64, 64)
        scene = "FloorPlan1"
        action_list = ["MoveAhead", "RotateLeft", "RotateRight", "LookUp", "LookDown"]
        time_limit = 10

    config = DummyConfig()
    env = AI2ThorEnv(config)

    print("AI2-THOR Controller参数：")
    print("Scene:", config.scene)
    print("Action List:", config.action_list)
    print("Image Size:", config.size)
    print("Max Steps:", config.time_limit)
    print("Action Space:", env.action_space)
    print("Observation Space:", env.observation_space)

    obs = env.reset()
    print("初始观测obs keys:", obs.keys())
    print("初始观测image shape:", obs["image"].shape)

    for i in range(5):
        action = env.action_space.sample()
        print(f"\nStep {i+1}: 执行动作 {config.action_list[action]}")
        obs, reward, done, info = env.step(action)
        print("观测image shape:", obs["image"].shape)
        print("reward:", reward, "done:", done)
        if done:
            print("Episode done, 重置环境")
            obs = env.reset()
import gym
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    env = gym.wrappers.Monitor(env, "LL_recordings")


model = DQN(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("deepq-lunarlander")


del model   # remove to demonstrate saving and loading

model = DQN.load("deepq-lunarlander")

state = env.reset()
while True:
    action, _states = model.predict(state)
    state, rewards, done, info = env.step(action)
    env.render()

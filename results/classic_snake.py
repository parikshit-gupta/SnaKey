import snake_RL_env
from DQN_agent import DQNagent
import gymnasium as gym
from gymnasium.wrappers import RecordVideo


import tensorflow as tf
import numpy as np
import time


training_period=50
num_episodes=1000
record_performance_every=50
min_e=.01
total_e_decay_cycles=3


env=gym.make("snake_RL_env/ClassicSnake-v0", render_mode="rgb_array")
env = RecordVideo(
    env,
    video_folder="Classic_snake_gym_vid",
    name_prefix="training",
    episode_trigger=lambda x: x % training_period == 0  
)



agent=DQNagent.agent(env.unwrapped.size, env.unwrapped.action_space.n)




# they store the avg_rew, min_rew and max_rew arrays after every e_decay_cycle
AVG_REW=[]
MIN_REW=[]
MAX_REW=[]

# these note the reward per episode and avg_rew, min_rew, max_rew every certain number of episodes
rewards=np.array([0], dtype=np.float64)
avg_rew=np.array([0], dtype=np.float64)
min_rew=np.array([0], dtype=np.float64)
max_rew=np.array([0], dtype=np.float64)

e_decay_cycle=1
for episode in range(num_episodes+1):
    obs1, info= env.reset()
    obs2, reward, terminate, truncate, info=env.step(0)
    
    st=np.array([obs2, obs1])   #(st+1, st)
    episode_reward=0    #cumulative reward obtained in a given episode
    done=0
    e=1  
    
    start_time=time.time()
    while not done:     #executing a single episode
        action=agent.e_greedy(st, e)
        obs, rt, terminate, truncate, info=env.step(action)
        st1=np.array([obs, st[0]])
        agent.update_replay_memory(st, action, rt, st1, terminate)
        done = terminate or truncate
        agent.train(done)
        
        st=st1
        episode_reward+=rt
        
    end_time=time.time()
    
    print(f"Time: {end_time - start_time:.2f}, Episode_reward: {episode_reward:.2f}")

    
    e=max(min_e, .999*e)    
    np.append(rewards, episode_reward)
    
    if (episode%record_performance_every==0):
        Z=rewards[-record_performance_every:]
        np.append(avg_rew, np.sum(Z)/Z.shape[0])
        np.append(min_rew, np.min(Z))
        np.append(max_rew, np.max(Z))
        
        # PENDING: code to log data in tensorboard using tf.summary
    
    # this helps our agent escape local minimas, if it ever gets stuck in one
    if(episode == e_decay_cycle*num_episodes // total_e_decay_cycles):
        e=1
        e_decay_cycle+=1
        AVG_REW.append(avg_rew)
        MIN_REW.append(min_rew)
        MAX_REW.append(max_rew)
        rewards=np.array([0], dtype=np.float64)
        avg_rew=np.array([0], dtype=np.float64)
        min_rew=np.array([0], dtype=np.float64)
        max_rew=np.array([0], dtype=np.float64)
        
        agent.model.save(f"checkpoints/dqn_episode_{episode}.keras")
        
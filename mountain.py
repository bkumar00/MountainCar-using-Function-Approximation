import numpy as np
import matplotlib.pyplot as plt
import gym
import itertools
import sys
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
if "../" not in sys.path:
    sys.path.append("../")

env = gym.envs.make("MountainCar-v0")



#collect samples to fit into featurizer
obs_exmps = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(obs_exmps)
featurizer = sklearn.pipeline.FeatureUnion([("rbf1", RBFSampler(gamma=5.0, n_components=100)),("rbf2", RBFSampler(gamma=2.0, n_components=100)),("rbf3", RBFSampler(gamma=1.0, n_components=100)),("rbf4", RBFSampler(gamma=0.5, n_components=100))])
featurizer.fit(scaler.transform(obs_exmps))


#estimator class is created
class Estimator():
    def __init__(self):
        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
            model.partial_fit([self.featurize_state(env.reset())], [0])
            self.models.append(model)

    def featurize_state(self, state):
        scaled = scaler.transform([state])
        featurized = featurizer.transform(scaled)
        return featurized[0]


    def predict(self, s, a=None):
        features = self.featurize_state(s)
        if not a:
            return np.array([m.predict([features])[0] for m in self.models])
        else:
            return self.models[a].predict([features])[0]
    def update(self, s, a, y):
        features = self.featurize_state(s)
        self.models[a].partial_fit([features], [y])



#standard epsilon greedy policy is used
def make_epsilon_greedy_policy(estimator, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


#main loop
def q_learning(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):
    
    #for plotting
    episode_lengths=np.zeros(num_episodes)
    episode_rewards=np.zeros(num_episodes)
    for i_episode in range(num_episodes):
         policy = make_epsilon_greedy_policy(estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)
         last_reward = episode_rewards[i_episode - 1]
         sys.stdout.flush()
         state = env.reset()
         next_action = None
         for t in itertools.count():
            if next_action is None:
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            else:
                action = next_action
            next_state, reward, done, _ = env.step(action)
            episode_rewards[i_episode] += reward
            episode_lengths[i_episode] = t
            q_values_next = estimator.predict(next_state)
            td_target = reward + discount_factor * np.max(q_values_next)
            estimator.update(state, action, td_target)
            print("\rStep {} @ Episode {}/{} ({})".format(t, i_episode + 1, num_episodes, last_reward), end="")
            if done:
                break

            state = next_state
    return episode_lengths,episode_rewards

estimator = Estimator()
epi_len,epi_reward = q_learning(env, estimator, 100, epsilon=0.0)


#visualisation
plt.plot(epi_len)
plt.ylabel("Length of each episode")
plt.xlabel("Number of episode")
plt.show()

plt.plot(epi_reward)
plt.ylabel("Reward of each episode")
plt.xlabel("Number of episode")
plt.show()

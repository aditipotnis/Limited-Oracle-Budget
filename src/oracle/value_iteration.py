import numpy as np


def value_iteration(env, gamma=0.99, theta=1e-8, max_iterations=10000):
    P = env.unwrapped.P
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    V = np.zeros(n_states)

    for _ in range(max_iterations):
        delta = 0.0
        for s in range(n_states):
            v = V[s]
            q_values = _compute_q(P, V, s, n_actions, gamma)
            V[s] = np.max(q_values)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break

    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        q_values = _compute_q(P, V, s, n_actions, gamma)
        policy[s] = np.argmax(q_values)

    return V, policy


def _compute_q(P, V, state, n_actions, gamma):
    q_values = np.zeros(n_actions)
    for a in range(n_actions):
        for prob, next_state, reward, done in P[state][a]:
            q_values[a] += prob * (reward + gamma * V[next_state] * (1.0 - float(done)))
    return q_values

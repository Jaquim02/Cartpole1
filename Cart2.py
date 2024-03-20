import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Crear el entorno CartPole
environment = gym.make("CartPole-v1", render_mode="rgb_array")
environment.action_space.seed(42)
np.random.seed(42)
observation, info = environment.reset(seed=42)

# Parámetros de control
EPISODES = 10000
CAPTURE_INTERVAL = 500
REWARD_UPDATE_INTERVAL = 500
LEARNING_COEFFICIENT = 500
EXPLORATION_COEFFICIENT = 500

# Inicializar estadísticas de recompensa
rewards_history = np.zeros(EPISODES)

# Definir los límites de discretización
NUM_BINS = (12, 12, 12, 12)
LOWER_BOUNDS = [environment.observation_space.low[0], -0.25, environment.observation_space.low[2], -np.radians(50)]
UPPER_BOUNDS = [environment.observation_space.high[0], 0.25, environment.observation_space.high[2], np.radians(50)]

# Función de discretización
def discretize_state(cart_position, cart_velocity, pole_angle, pole_velocity):
    def discretize_value(value, min_val, max_val, num_bins):
        value = max(min_val, min(value, max_val))
        bin_width = (max_val - min_val) / num_bins
        bin_index = int((value - min_val) / bin_width)
        return min(bin_index, num_bins - 1)

    cart_position_index = discretize_value(cart_position, LOWER_BOUNDS[0], UPPER_BOUNDS[0], NUM_BINS[0])
    cart_velocity_index = discretize_value(cart_velocity, LOWER_BOUNDS[1], UPPER_BOUNDS[1], NUM_BINS[1])
    pole_angle_index = discretize_value(pole_angle, LOWER_BOUNDS[2], UPPER_BOUNDS[2], NUM_BINS[2])
    pole_velocity_index = discretize_value(pole_velocity, LOWER_BOUNDS[3], UPPER_BOUNDS[3], NUM_BINS[3])

    return (cart_position_index, cart_velocity_index, pole_angle_index, pole_velocity_index)

# Función para elegir una acción basada en la política
def choose_action(state, q_table):
    return np.argmax(q_table[state])

# Tasa de exploración
def exploration_rate(episode, min_rate=0.01):
    return max(min_rate, min(1, 1.0 - np.log10((episode + 1) / EXPLORATION_COEFFICIENT)))

# Tasa de aprendizaje
def learning_rate(episode, min_rate=0.01):
    return max(min_rate, min(1.0, 1.0 - np.log10((episode + 1) / LEARNING_COEFFICIENT)))

# Calcular el nuevo valor Q
def update_q_value(reward, new_state, discount_factor=1):
    future_optimal_value = np.max(Q_TABLE[new_state])
    learned_value = reward + discount_factor * future_optimal_value
    return learned_value

# Inicializar la tabla Q
Q_TABLE = np.zeros(NUM_BINS + (environment.action_space.n,))

# Bucle de entrenamiento
for episode in range(EPISODES):
    observation, info = environment.reset()
    current_state = discretize_state(*observation)
    total_reward = 0

    while True:
        # Elegir acción basada en la política
        action = choose_action(current_state, Q_TABLE)
        # Exploración aleatoria
        if np.random.random() < exploration_rate(episode):
            action = environment.action_space.sample()
        
        # Ejecutar acción y observar resultado
        observation, reward, done, _, _ = environment.step(action)
        new_state = discretize_state(*observation)

        # Actualizar la tabla Q
        lr = learning_rate(episode)
        learned_value = update_q_value(reward, new_state)
        old_value = Q_TABLE[current_state][action]
        Q_TABLE[current_state][action] = (1 - lr) * old_value + lr * learned_value

        # Actualizar estado y recompensa total
        current_state = new_state
        total_reward += reward

        if done:
            break

    # Registrar recompensa del episodio
    rewards_history[episode] = total_reward

    # Imprimir información periódicamente
    if episode % REWARD_UPDATE_INTERVAL == 0:
        avg_reward = np.mean(rewards_history[max(0, episode - REWARD_UPDATE_INTERVAL):episode + 1])
        min_reward = np.min(rewards_history[max(0, episode - REWARD_UPDATE_INTERVAL):episode + 1])
        max_reward = np.max(rewards_history[max(0, episode - REWARD_UPDATE_INTERVAL):episode + 1])
        print(f"Episode: {episode}, Avg Reward: {avg_reward}, Min Reward: {min_reward}, Max Reward: {max_reward}")

# Define a basic policy function
def basic_policy(observation):
    # Example policy: if angle is positive, move right, otherwise move left
    return 1 if observation[2] > 0 else 0

# Function to update the scene for animation
def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

# Function to create and display animation of an episode
def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    plt.show()
    plt.close()

# Function to run and display one episode
def show_one_episode(policy, n_max_steps=200, seed=42):
    frames = []
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    np.random.seed(seed)
    obs, info = env.reset(seed=seed)
    for step in range(n_max_steps):
        frames.append(env.render())
        action = policy(obs)
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            break
    env.close()
    plot_animation(frames)

# Call show_one_episode function to display an episode
show_one_episode(basic_policy)

# Close the environment
environment.close()
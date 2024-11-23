# main.py

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import gym
from gym import spaces
import tensorflow as tf
from collections import deque
import random
import openai
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set 'OPENAI_API_KEY' in your .env file.")

def get_preprocess_data(ticker, start_date, end_date):
    """
    Fetches and preprocesses stock data.
    Args:
        ticker (str): Stock symbol
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
    Returns:
        pandas.DataFrame: Processed stock data
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        data.dropna(inplace=True)
        data.reset_index(inplace=True)
        os.makedirs('data', exist_ok=True)
        data.to_csv(f'data/{ticker}_data.csv', index=False)
        print(f"Data for {ticker} saved successfully.")
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

class TradingEnv(gym.Env):
    """
    Custom OpenAI Gym environment for stock trading.
    Implements basic trading mechanics with a simple reward system.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.total_steps = len(df) - 1
        self.current_step = 0

        # Define action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Define observation space: OHLCV + Technical Indicators
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self.reset()

    def reset(self):
        """Resets the trading environment to initial state."""
        self.balance = 10000  # Initial capital
        self.max_balance = 10000
        self.shares_held = 0
        self.net_worth = 10000
        self.max_net_worth = 10000
        self.current_step = 0
        self.trade_history = []
        return self._next_observation()

    def _next_observation(self):
        """
        Constructs the next observation from the data.
        Returns:
            numpy.array: Current market state observation
        """
        obs = self.df.loc[self.current_step, ['Open', 'High', 'Low', 'Close', 'Volume']].values

        # Calculate 5-day Simple Moving Average
        window = 5
        if self.current_step >= window:
            sma = self.df['Close'][self.current_step - window:self.current_step].mean()
        else:
            sma = self.df['Close'][:self.current_step + 1].mean()

        obs = np.append(obs, sma)
        return obs.astype(np.float32)

    def step(self, action):
        """
        Execute one trading step in the environment.
        Args:
            action (int): Trading action to take (0: Hold, 1: Buy, 2: Sell)
        Returns:
            tuple: (observation, reward, done, info)
        """
        self._take_action(action)

        self.current_step += 1
        if self.current_step > self.total_steps:
            # End episode if we've reached the end of the data
            done = True
            obs = self._next_observation()
            return obs, 0, done, {}
        else:
            done = False

        # Calculate reward based on change in net worth
        reward = self.net_worth - self.max_net_worth
        self.max_net_worth = max(self.max_net_worth, self.net_worth)

        # Check if net worth has dropped below zero
        if self.net_worth <= 0:
            done = True
            reward = -1  # Penalize for going bankrupt

        obs = self._next_observation()
        return obs, reward, done, {}

    def _take_action(self, action):
        """
        Executes the trading action in the environment.
        Args:
            action (int): Trading action to take
        """
        current_price = self.df.loc[self.current_step, 'Close']

        if action == 1:  # Buy
            shares_can_buy = self.balance // current_price
            if shares_can_buy > 0:
                self.balance -= shares_can_buy * current_price
                self.shares_held += shares_can_buy
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'buy',
                    'shares': shares_can_buy,
                    'price': current_price
                })

        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'sell',
                    'shares': self.shares_held,
                    'price': current_price
                })
                self.shares_held = 0

        self.net_worth = self.balance + self.shares_held * current_price

class DQNAgent:
    """
    Deep Q-Network agent for trading decisions.
    Implements epsilon-greedy policy for exploration vs exploitation.
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        """
        Builds neural network model for Q-learning.
        Returns:
            tensorflow.keras.Model: Compiled neural network model
        """
        model = tf.keras.models.Sequential([
            tf.keras.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(
            loss='mse',
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        """Stores experience tuple in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Chooses action based on epsilon-greedy policy.
        Args:
            state (numpy.array): Current state observation
        Returns:
            int: Selected action
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        """
        Trains the model using experience replay.
        Args:
            batch_size (int): Size of training batch
        """
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            state = np.array(state).reshape(1, self.state_size)
            next_state = np.array(next_state).reshape(1, self.state_size)
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        # Decay the exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def get_basic_explanation(state, action, current_price, prev_price):
    """
    Generates a basic explanation for trading decisions without using API.
    """
    explanation = "Decision based on: "
    
    # Calculate basic indicators
    price_change = ((current_price - prev_price) / prev_price) * 100 if prev_price != 0 else 0
    sma = state[5]
    
    if action == 0:  # Hold
        explanation += "Market conditions unclear. Holding position."
    elif action == 1:  # Buy
        if current_price < sma:
            explanation += "Price below moving average - potential buying opportunity."
        else:
            explanation += "Upward momentum detected."
    else:  # Sell
        if current_price > sma:
            explanation += "Price above moving average - taking profits."
        else:
            explanation += "Downward trend detected - minimizing losses."
            
    return explanation

def get_ai_explanation(state, action, trade_count):
    """
    Gets AI-powered explanation for significant trades.
    Only uses API every 10 trades to minimize costs.
    """
    if trade_count % 10 != 0:  # Only get AI explanation every 10 trades
        return None

    try:
        state_info = (
            f"Open: {state[0]:.2f}, High: {state[1]:.2f}, "
            f"Low: {state[2]:.2f}, Close: {state[3]:.2f}, "
            f"Volume: {state[4]:.2f}, SMA: {state[5]:.2f}"
        )

        prompt = (
            f"As a financial analyst, briefly explain in one sentence why "
            f"the trading agent decided to {['Hold', 'Buy', 'Sell'][action]} "
            f"based on: {state_info}"
        )

        # Use ChatCompletion API with 'gpt-3.5-turbo'
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": "system", "content": "You are a financial analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,  # Reduced token count
            temperature=0.7,
            n=1,
            stop=None
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error getting AI explanation: {e}")
        return None

def train_agent(env, agent, episodes=5, batch_size=32):
    """
    Trains the DQN agent.
    Args:
        env (TradingEnv): Trading environment
        agent (DQNAgent): DQN agent
        episodes (int): Number of training episodes
        batch_size (int): Size of training batch
    """
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, agent.state_size]).astype(np.float32)

        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, agent.state_size]).astype(np.float32)

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                print(f"Episode {e+1}/{episodes} finished after {time+1} timesteps")
                break

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        # Save model after each episode
        os.makedirs('models', exist_ok=True)
        agent.model.save('models/trading_model.h5')
        print(f"Episode {e+1}/{episodes} - Net Worth: {env.net_worth:.2f}")

def main():
    """Main function to run the trading bot."""
    # Get user inputs
    ticker = input("Enter Stock Ticker (default 'AAPL'): ") or "AAPL"
    start_date = input("Enter Start Date (YYYY-MM-DD, default '2019-01-01'): ") or "2019-01-01"
    end_date = input("Enter End Date (YYYY-MM-DD, default '2020-12-31'): ") or "2020-12-31"

    # Initialize data and environment
    data = get_preprocess_data(ticker, start_date, end_date)
    if data is None:
        print("Failed to get data. Exiting.")
        return

    df = data  # Use the data returned from get_preprocess_data to avoid re-reading from CSV
    env = TradingEnv(df)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    # Load or train model
    model_path = 'models/trading_model.h5'
    if os.path.exists(model_path):
        agent.model.load_weights(model_path)
        print("Model loaded.")
    else:
        print("Training new model...")
        train_agent(env, agent, episodes=5)  # Reduced episodes to minimize training time

    # Run trading simulation
    state = env.reset()
    state = np.reshape(state, [1, agent.state_size]).astype(np.float32)
    total_reward = 0
    net_worths = [env.net_worth]
    trade_count = 0

    for i in range(len(df) - 1):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        # Get current and previous prices
        current_price = df.loc[env.current_step, 'Close']
        prev_price = df.loc[env.current_step - 1, 'Close'] if env.current_step > 0 else current_price

        # Get basic explanation for all trades
        basic_explanation = get_basic_explanation(state[0], action, current_price, prev_price)

        # Get AI explanation for significant trades
        ai_explanation = None
        if action != 0:  # Only count actual trades
            trade_count += 1
            ai_explanation = get_ai_explanation(state[0], action, trade_count)

        # Display results
        print(f"\nStep {env.current_step}")
        print(f"Action: {['Hold', 'Buy', 'Sell'][action]}")
        print(f"Basic Analysis: {basic_explanation}")
        if ai_explanation:
            print(f"AI Analysis: {ai_explanation}")
        print(f"Net Worth: {env.net_worth:.2f}")
        print("---")

        state = np.reshape(next_state, [1, agent.state_size]).astype(np.float32)
        net_worths.append(env.net_worth)

        if done:
            print("Trading session ended.")
            break

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(net_worths)
    plt.title(f'Trading Bot Performance - {ticker}')
    plt.xlabel('Time Steps')
    plt.ylabel('Net Worth ($)')
    plt.grid(True)
    plt.show()

    print(f"Final Net Worth: ${env.net_worth:.2f}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Total Trades: {trade_count}")

if __name__ == '__main__':
    main()

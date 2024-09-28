# Q-Learning Pathfinding Project

## Overview

This project implements a **Q-Learning** algorithm to solve pathfinding tasks in a custom grid environment. The environment includes various zones, obstacles, item pickup points, and drop-off locations. The Q-Learning agent learns to navigate the environment and optimize its performance over multiple episodes.

The project also includes tools for:
- **Training** the Q-Learning agent with configurable hyperparameters.
- **Saving and loading** Q-tables for future predictions.
- **Visualizing** training rewards and performance metrics.
- **Analyzing** the agent's learned Q-table for decision-making insights.

## Features

- **Custom Grid Environment**: The agent navigates a grid, avoiding obstacles and completing tasks such as item pickup and drop-off.
- **Q-Learning Algorithm**: Implements Q-Learning to train the agent over multiple iterations.
- **Save/Load Q-Table**: Easily save and load trained models.
- **Reward Visualization**: Graphically display rewards during training to monitor learning performance.
- **Q-Table Analysis**: Analyze the Q-Table after training for insights into the decision-making process.

## Files

- **`QLearning.py`**: Contains the core implementation of the Q-Learning algorithm, including training, evaluation, and Q-table management functionalities.
- **`Games.py`**: Defines the grid environment, including parameters for obstacles, item pickups, and agent actions.
- **`main.py`**: The main entry point to run training, predictions, and analysis of the Q-Learning agent. Includes options for customizing grid sizes, hyperparameters, and visualization.

## How to Run

1. **Install Requirements**:

   Ensure you have the necessary dependencies installed in requirements.txt

2. **Train the Q-Learning Agent**:
   
   You can run the training process using the `main.py` script. By default, the agent will train on a 10x10 grid with predefined item locations and blocked zones. Hyperparameters such as the number of iterations, epsilon, and gamma are set within the code, but can be modified in the script.

3. **Visualize Rewards**:
   
   After training, the project will output reward visualizations, allowing you to see how the agent’s performance improved over time.

4. **Q-Table Analysis**:
   
   You can analyze the learned Q-Table by running the analysis section in `main.py`:

5. **Load and Predict**:
   
   Use a saved Q-table to make predictions in the environment without retraining.

## Hyperparameters

You can adjust the following hyperparameters in `main.py` or in `QLearning.py`:

- **`gamma`**: Discount factor for future rewards.
- **`epsilon`**: Exploration-exploitation tradeoff parameter.
- **`alpha`**: Learning rate for updating Q-values.

## Output

- **Reward Plot**: A plot of rewards over training episodes.
- **Q-Table File**: The Q-Table is saved as a `.joblib` file for future use.
- **Scatter Plots**: 2D and 3D scatter plots of agent performance metrics are also saved.

## Contributing

Contributions to improve the Q-Learning agent, add more features, or extend the environment are welcome. Feel free to submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
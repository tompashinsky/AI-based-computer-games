# Final Project | AI-based Computer Games 🎮
This project was submitted by Netanel Druyan and Tom Pashinsky

## Abstract 📋
The project focuses on developing an interactive gaming environment that incorporates elements of Artificial Intelligence (AI). The environment includes a series of games in which the player competes against a smart digital opponent trained using machine learning algorithms. The main goal of the project is to demonstrate how machine learning algorithms can be applied in dynamic, real-time environments, creating a challenging and ever-changing gameplay experience. The environment is designed to be modular, easy to expand, and capable of integrating additional games or adjusting difficulty levels in the future. The technologies used in the project have wide-ranging applications, including game development in Python and training machine learning models. The implementation of the project involves several stages: defining the structure of the environment, specifying each game and its opponent, building a learning algorithm for each opponent, and developing the user interface. At each stage, the performance of the digital opponent will be evaluated and adjusted to improve both the user experience and the opponent's learning ability. <br>

--------------

## Project Deliverables

### 1. Home Screen 📱
This is the main screen of the gaming environment. At the top of the screen appears the name of the environment (NTGames—a name that might change in the future), and in the center, buttons with the names of the available games. In terms of design, we created a central function called ***main_menu*** that draws the UI components of the home screen and handles button clicks. Additionally, a general function called ***game_controller*** manages transitions between the home screen and the games themselves, and vice versa.

### 2. Tic Tac Toe Game ✖️⭕
On the home screen, there's a button for the Tic Tac Toe game. Clicking the button leads to a screen where the user chooses which shape to play: X or O. If the user chooses X, they start the game. Otherwise, the computer makes the first move. At the end of each game, a popup appears announcing the winner (or a draw), along with two buttons: **“Restart”** (to start a new game) or **“Quit”** (to return to the home screen).

#### AI Opponent Development
Our Tic Tac Toe opponent is based on the MCTS (Monte Carlo Tree Search) model. This machine learning algorithm combines the precision of search trees with the efficiency of random choice. Each node in the tree represents a current game state — a two-dimensional array that stores several elements: player identifiers (the human player is represented as number 1, and the AI player as number 2), each player’s chosen shape (X or O), and the specific cell they selected.

### 3. Head Soccer Game ⚽
On the home screen of the gaming environment, there is a button for the Head Soccer game. Clicking the button launches the game, and the timer starts running (each match lasts 3 minutes). At the end of each match, a popup window announces the winner (or a draw), along with two buttons: **“Restart”** (to start a new game) or **“Quit”** (to return to the home screen).
Most of the game’s functionality is centralized in the main function ***run_head_soccer***, which manages the entire gameplay process. At the beginning of this function, the main objects are initialized: the two players (using the Player class) and the ball (using the Ball class). Then, a game loop is run, and during each iteration, several main actions occur: processing user input (movement, jumping, and kicking), updating the states of the players and the ball, checking for collisions (between players and the ball, and between the ball and the goals or crossbars), detecting valid goals, and displaying a “GOAL!” message using the ***show_goal_message*** function. Additionally, the game screen continuously displays time and score updates. When the time runs out, the ***show_game_over_screen*** function is called, presenting the final result and offering the options to restart or return to the menu.

### 4. Bubbles Game 🔴🔵🟡
On the home screen of the gaming environment, there is a button for the Bubbles game. This game is a unique variation of the classic bubble shooter game. Players are positioned on opposite sides of the screen (left and right), while bubbles appear in the center, separated by a shared wall. When one player pops bubbles, the shared wall moves toward the other player, bringing the bubbles closer to them and increasing the risk of losing.
The algorithm is designed so that each player has their own matrix of bubbles, arranged in a honeycomb structure and managed separately for each player. The algorithm includes methods for detecting collisions and identifying the bubble closest to the one currently being launched, ensuring proper connection. At the end of each match, a popup window appears, announcing the winner (or a draw), along with two buttons: **“Restart”** (to start a new game) or **“Quit”** (to return to the home screen).



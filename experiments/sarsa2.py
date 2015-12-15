from game_manager import GameManager
from agents import Sarsa2Agent
from . import get_results_path

results_path = get_results_path(__name__)

agent = Sarsa2Agent(n_frames_per_action=1,
                    trace_type='accumulating', 
                    learning_rate=0.001, 
                    discount=0.999, 
                    path='fig',
                    capture=True,
                    lambda_v=0.9)

gm = GameManager("pong.bin",
                 agent, results_path,
                 remove_old_results_dir=True,
                 n_episodes=10,
                 n_epochs=6)

gm.run()

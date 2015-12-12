from game_manager import GameManager
from agents import SmartAgent
import os

experiment = os.path.basename(__file__).rpartition('.')[0]
results_path = os.path.join('results', experiment)

agent = SmartAgent()

gm = GameManager("pong.bin",
                 agent, results_path,
                 n_episodes=500,
                 remove_old_results_dir=True, use_minimal_action_set=True, 
                 min_time_between_frames=0.0)

gm.run()

from game_manager import GameManager
from agents import SarsaAgent
import os

experiment = os.path.basename(__file__).rpartition('.')[0]
results_path = os.path.join('results', experiment)

agent = SarsaAgent(n_frames_per_action=10,
                   trace_type='replacing', 
                   learning_rate=0.01, 
                   discount=0.99, 
                   lambda_v=0.5)

gm = GameManager("pong.bin",
                 agent, results_path,
                 remove_old_results_dir=True, use_minimal_action_set=True,
                 n_episodes=500,
                 # min_time_between_frames=0.000001,
                 min_time_between_frames=0)

gm.run()

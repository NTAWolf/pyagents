from game_manager import GameManager
from agents import DevelopmentAgent as Ag
import os

experiment = os.path.basename(__file__).rpartition('.')[0]
results_path = os.path.join('results', experiment)

agent = Ag()
gm = GameManager("pong.bin", agent, results_path,
                 n_episodes=10,
                 remove_old_results_dir=True, use_minimal_action_set=True)
gm.run()

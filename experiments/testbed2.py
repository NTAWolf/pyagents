from game_manager import GameManager
from agents import ActionChainAgent
import os

experiment = os.path.basename(__file__).rpartition('.')[0]
results_path = os.path.join('results', experiment)

agent = ActionChainAgent(3)
gm = GameManager("space_invaders.bin", agent, results_path,
                 n_episodes=2,
                 remove_old_results_dir=True, use_minimal_action_set=True)
gm.run()

from game_manager import GameManager
from agents import UniformRandomAgent
import os

experiment = os.path.basename(__file__).rpartition('.')[0]
results_path = os.path.join('results', experiment)

agent = UniformRandomAgent()
gm = GameManager("pong.bin", agent, results_path,
                 n_epochs=3, n_episodes=10,
                 remove_old_results_dir=True, use_minimal_action_set=True)
gm.run()

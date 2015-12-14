from game_manager import GameManager
from agents import SmartAgent
from . import get_results_path

results_path = get_results_path(__name__)

agent = SmartAgent()

gm = GameManager("pong.bin",
                 agent, results_path,
                 n_epochs=5, n_episodes=8,
                 remove_old_results_dir=True, use_minimal_action_set=True, 
                 min_time_between_frames=0.0)

gm.run()

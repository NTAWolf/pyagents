from game_manager import GameManager
from agents import ActionChainAgent
from . import get_results_path

results_path = get_results_path(__name__)

agent = ActionChainAgent(3)
gm = GameManager("space_invaders.bin", agent, results_path,
                 n_episodes=2,
                 remove_old_results_dir=True, use_minimal_action_set=True)
gm.run()

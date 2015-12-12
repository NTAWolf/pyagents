from game_manager import GameManager
from agents import DevelopmentAgent as Ag
from . import get_results_path

results_path = get_results_path(__name__)

agent = Ag()
gm = GameManager("pong.bin", agent, results_path,
                 n_episodes=10,
                 remove_old_results_dir=True, use_minimal_action_set=True)
gm.run()

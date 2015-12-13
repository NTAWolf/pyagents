from game_manager import GameManager
from agents import UniformRandomAgent
from . import get_results_path

results_path = get_results_path(__name__)

agent = UniformRandomAgent()
gm = GameManager("pong.bin", agent, results_path,
                 n_epochs=3, n_episodes=10,
                 remove_old_results_dir=True, use_minimal_action_set=True)
gm.run()

from game_manager import GameManager
from agents import UniformRandomAgent

agent = UniformRandomAgent()
gm = GameManager("pong.bin", agent, 'experiments/rvresults',
                 n_episodes=10,
                 remove_old_results_dir=True, use_minimal_action_set=True)
gm.run()

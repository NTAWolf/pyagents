from game_manager import GameManager
from agents import DevelopmentAgent as Ag

agent = Ag()
gm = GameManager("pong.bin", agent, 'results/testbed2',
                 n_episodes=10,
                 remove_old_results_dir=True, use_minimal_action_set=True)
gm.run()

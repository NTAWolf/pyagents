from game_manager import GameManager
from agents import ActionChainAgent

agent = ActionChainAgent(3)
gm = GameManager("space_invaders.bin", agent, 'results/testbed2',
                 remove_old_results_dir=True, use_minimal_action_set=True)
gm.run(n_episodes=2)

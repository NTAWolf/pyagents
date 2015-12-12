from game_manager import GameManager
from agents import SmartAgent

agent = SmartAgent()

gm = GameManager("pong.bin",
                 agent, 'results/smart',
                 remove_old_results_dir=True, use_minimal_action_set=True, 
                 min_time_between_frames=0.0)

gm.run(n_episodes=500)

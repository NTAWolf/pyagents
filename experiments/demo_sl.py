from game_manager import GameManager
from agents import SLAgent

agent = SLAgent(n_frames_per_action=1)

gm = GameManager("pong.bin",
                 agent, 'results/testbed3',
                 remove_old_results_dir=True, use_minimal_action_set=True, 
                 visualise=None, min_time_between_frames=0.000001)
                 # visualise='rgb')

gm.run(n_episodes=500)

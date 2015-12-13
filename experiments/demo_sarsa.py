from game_manager import GameManager
from agents import SarsaAgent
from . import get_results_path

results_path = get_results_path(__name__)

agent = SarsaAgent(n_frames_per_action=1,
                   trace_type='accumulating', 
                   learning_rate=0.001, 
                   discount=0.999, 
                   lambda_v=0.9)

gm = GameManager("pong.bin",
                 agent, results_path,
                 remove_old_results_dir=True, use_minimal_action_set=True,
                 n_episodes=10,
                 # min_time_between_frames=0.000001,
                 min_time_between_frames=0)

gm.run()

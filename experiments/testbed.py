from basis import GameManager
from agents.uniform_random import UniformRandomAgent

agent = UniformRandomAgent()
gm = GameManager("space_invaders.bin", agent, 'experiments/rvresults', remove_old_results_dir=True, use_minimal_action_set=True, visualise='ram')
gm.run(n_episodes=10)

from basis import GameManager
from agents.action_chain import ActionChainAgent

agent = ActionChainAgent(10)
gm = GameManager("space_invaders.bin", agent, 'experiments/acaresults',
                 remove_old_results_dir=True, use_minimal_action_set=True, visualise=None)#'rgb')
gm.run(n_episodes=1000)

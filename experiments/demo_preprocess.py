from game_manager import GameManager
from agents import UniformRandomAgent
from agents.dl import Preprocessor

from time import sleep

agent = UniformRandomAgent()

def visfunc(state_methods):
    update_n_frames = 2
    pp = Preprocessor((128,128), update_n_frames, 1)
    counter = [update_n_frames]
    cache = [None]

    def func():
        if counter[0]%update_n_frames != 0:
            counter[0] += 1
            return cache[0]
        
        counter[0] = 1
        cache[0] = pp.trace(state_methods)
        return cache[0]
    return func


gm = GameManager("space_invaders.bin", agent, 'results/testbed3',
                 remove_old_results_dir=True, use_minimal_action_set=True, visualise=visfunc)
gm.run(n_frames=30000)

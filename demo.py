from basis import GameManager
from agents.uniform_random import UniformRandomAgent

agent = UniformRandomAgent()
gm = GameManager("alien.bin", agent, 'demo_results')

gm.run(n_frames=10000)

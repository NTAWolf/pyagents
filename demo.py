from game_manager import GameManager
from agents import UniformRandomAgent

agent = UniformRandomAgent()
gm = GameManager("alien.bin", agent, 'demo_results')

gm.run(n_frames=10000)

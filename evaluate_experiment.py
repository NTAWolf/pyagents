#!/usr/bin/env python

from datetime import timedelta
import os
import re
import subprocess

import pandas as pd


# def add_to_hall_of_fame(log):


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Experiment evaluator for pyagents.')
    parser.add_argument('logfile',
                        help=('The path to the logfile of the experiment you '
                              'want to evaluate.')
                        )

    logfile = parser.parse_args().logfile
    print "logfile", logfile
    figfile = os.path.join(os.path.dirname(logfile), 'episode_stats.png')
    print "figfile", figfile


    with open(logfile, 'r') as f:
        log = f.readlines()

    episode_stats_re = re.compile(
        '^episode: Ended with total reward (\d+) after (.*)$')

    episode_stats = [m.groups()
                     for m in map(episode_stats_re.match, log) if m != None]
    episode_stats = [(int(reward), int(duration))
                     for reward, duration in episode_stats]
    episode_stats = pd.DataFrame(episode_stats, columns=[
                                 "total_reward", "duration"])
    episode_stats['total_reward_win20'] = pd.rolling_mean(episode_stats.total_reward, 20)
    episode_stats['total_reward_win50'] = pd.rolling_mean(episode_stats.total_reward, 50)

    ax = episode_stats[['total_reward', 'total_reward_win20', 'total_reward_win50']].plot()
    fig = ax.get_figure()
    fig.savefig(figfile)

    subprocess.call(['open', figfile])


if __name__ == '__main__':
    main()
else:
    raise ImportError("The evaluate_experiment module can only be run as main")

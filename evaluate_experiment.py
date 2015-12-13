#!/usr/bin/env python

from __future__ import print_function
import argparse

import os
import shutil
import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn # Makes plots look prettier

import experiments

class Evaluator(object):

    # 1000 forms of initialization
    def __init__(s, results_dir):
        s.results_dir = results_dir
        s.init_stats()
        s.init_settings()
        s.init_eval_path()
        s.init_q_e()

        s.evaluate()

    def init_stats(s):
        stats_path = s.get_path('stats.log')
        stats = pd.read_csv(stats_path)
        s.stats = stats

    def init_settings(s):
        settings_path = s.get_path('settings')

        with open(settings_path, 'r') as f:
            settings = json.load(f)

        s.settings = settings

    def init_q_e(s):
        q_e_path = s.get_path('q_e.csv')
        try:
            s.q_e = pd.read_csv(q_e_path)
        except IOError:
            print ("Cannot find {}. Will not evaluate on Q "
                   "and e values.".format(q_e_path))
            s.q_e = None
            

    def init_eval_path(s):
        eval_path = s.get_path('evaluation')
        if os.path.exists(eval_path):
            shutil.rmtree(eval_path)
        os.makedirs(eval_path)
        s.eval_path = eval_path

    def get_path(s, name):
        return os.path.join(s.results_dir, name)

    # Actual evaluation
    def evaluate(s):
        s.plot_mean_reward_per_episode()
        s.plot_q_value()

    def plot_mean_reward_per_episode(s):
        episode_mean = s.stats.groupby('episode').mean()
        episode_mean.total_reward.plot(style=['o'])
        plt.ylabel('Mean total reward over {} epochs'.format(
                        len(s.stats.epoch.unique())))
        plt.title('{}'.format(s.settings['game_name']))

        s.expand_plot_lims()
        s.savefig('mean_reward_per_episode.png')

    def plot_q_value(s):
        if s.q_e is None:
            return
        cols = [c for c in s.q_e.columns if c.startswith('q')]
        q_vals = s.q_e[cols]

        q_vals.plot(style=['-'])
        plt.ylabel('Q-value')
        plt.title('Q-values for {}'.format(s.settings['game_name']))
        s.expand_plot_lims()
        s.savefig('q_vals.png')

    # Plot utils
    def expand_plot_lims(s, d=1):
        plt.xlim(s.expand(plt.xlim()))
        plt.ylim(s.expand(plt.ylim()))

    def expand(s, vals):
        ran = max(vals) - min(vals)
        delta = .05 * ran
        return (vals[0]-delta, vals[1]+delta)

    def savefig(s, filename):
        fig_path = os.path.join(s.eval_path, filename)
        plt.savefig(fig_path)
        print("Saved fig in {}".format(fig_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Experiment evaluator for pyagents.')
    parser.add_argument('experiment',
                        help=('Name of experiment to evaluate. Automatically '
                              'uses the results of the very latest run of '
                              'the experiment.'),
                        default='HELP_ME_I_FORGOT_TO_GIVE_IT_AN_EXPERIMENT',
                        nargs='?'
                        )

    experiment = parser.parse_args().experiment
    if experiments.has(experiment):
        results_dir = experiments.get_newest_results_path(experiment)
        Evaluator(results_dir)
        # evaluate(results_dir)
else:
    raise ImportError("The evaluate_experiment module can only be run as main")

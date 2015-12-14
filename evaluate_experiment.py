#!/usr/bin/env python

import argparse

import os
import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Makes plots look prettier

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
            # Clear the directory
            files = os.listdir(eval_path)
            for f in files:
                os.remove(os.path.join(eval_path,f))
        else:
            os.makedirs(eval_path)
        s.eval_path = eval_path

    def get_path(s, name):
        return os.path.join(s.results_dir, name)

    # Actual evaluation
    def evaluate(s):
        s.plot_mean_reward_per_episode()
        s.plot_reward_per_episode()
        s.plot_q_value()

    def plot_mean_reward_per_episode(s):
        episode_mean = s.stats.groupby('episode').mean()
        episode_mean.total_reward.plot(style=['o-'])

        plt.ylabel('Mean total reward over {} epochs'.format(
                        len(s.stats.epoch.unique())))
        plt.title('{}'.format(s.settings['game_name']))

        s.expand_plot_lims()
        s.savefig('mean_reward_per_episode.png')

    def plot_reward_per_episode(s):
        # etr = s.stats[['episode', 'total_reward']]
        # fig = etr.plot(x='episode',y='total_reward', style='o', legend='False')
        sns.boxplot(x="episode", y="total_reward", data=s.stats)
        plt.ylabel('Total reward in episode')

        s.expand_plot_lims()
        s.savefig('reward_per_episode.png')

    def plot_q_value(s):
        if s.q_e is None:
            return

        s.q_e = s.downsample(s.q_e, 3000)

        cols = [c for c in s.q_e.columns if c.startswith('q')]
        q_vals = s.q_e[cols]

        q_vals.plot(style=['-'])
        plt.ylabel('Q-value')
        plt.title('Q-values for {}'.format(s.settings['game_name']))

        s.expand_plot_lims()

        # TODO draw those vertical lines
        # # Draw a vertical line for each new episode
        # episode_val = s.q_e.episode.values
        # # True where the episode value changes, False elsewhere
        # ec = (episode_val[1:] + -1 * episode_val[:-1]).astype(bool)
        # # Include first frame as a new episode
        # ec = [True] + ec + [False]
        # # Get indices of episode changes
        # ec = q_vals.index.values[ec]
        # s.vertical_lines(ec)

        s.savefig('q_vals.png')

    # Data utils
    def downsample(s, df, n_rows):
        group_len = len(df) / n_rows
        if group_len <= 1:
            return df
        return df.groupby(lambda x: x/group_len).mean()

    # Plot utils
    def expand_plot_lims(s, d=1):
        plt.xlim(s.expand(plt.xlim()))
        plt.ylim(s.expand(plt.ylim()))

    def expand(s, vals):
        ran = max(vals) - min(vals)
        delta = .05 * ran
        return (vals[0]-delta, vals[1]+delta)

    def vertical_lines(s, x_vals):
        y1, y2 = plt.ylim()
        for x in x_vals:
            plt.plot((x,y1), (x,y2), 'k-')

    def savefig(s, filename):
        plt.tight_layout()
        fig_path = os.path.join(s.eval_path, filename)
        plt.savefig(fig_path)
        print "Saved fig in {}".format(fig_path)


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

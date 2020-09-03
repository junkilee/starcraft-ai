import sc2ai
from sc2ai.spinup.config import DEFAULT_BACKEND
from sc2ai.spinup.utils.run_utils import ExperimentGrid
from sc2ai.spinup.utils.serialization_utils import convert_json
import argparse
import gym
import json
import os
import subprocess
import sys
import os.path as osp
import string
import torch
from copy import deepcopy
from textwrap import dedent


RUN_KEYS = ['num_cpu', 'data_dir', 'datestamp']

SUBSTITUTIONS = {'env': 'env_name',
                 'hid': 'ac_kwargs:hidden_sizes',
                 'act': 'ac_kwargs:activation',
                 'cpu': 'num_cpu',
                 'dt': 'datestamp'}

MPI_COMPATIBLE_ALGOS = ['vpg', 'trpo', 'ppo']

BASE_ALGO_NAMES = ['vpg', 'trpo', 'ppo', 'ddpg', 'td3', 'sac']


def add_with_backends(algo_list):
    algo_list_with_backends = deepcopy(algo_list)
    for algo in algo_list:
        algo_list_with_backends += [algo + '_tf1', algo + '_pytorch']
    return algo_list_with_backends


def friendly_err(err_msg):
    return '\n\n' + err_msg + '\n\n'


def parse_and_execute_grid_search(cmd, args):
    if cmd in BASE_ALGO_NAMES:
        backend = DEFAULT_BACKEND[cmd]
        print('\n\nUsing default backend (%s) for %s.\n' % (backend, cmd))
        cmd = cmd + '_' + backend

    algo = eval('sc2ai.spinup.' + cmd)

    valid_help = ['--help', '-h', 'help']
    if any([arg in valid_help for arg in args]):
        print('\n\nShowing docstring for spinup.' + cmd + ':\n')
        print(algo.__doc__)
        sys.exit()

    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    arg_dict = dict()
    for i, arg in enumerate(args):
        assert i > 0 or '--' in arg, friendly_err("You didn't specify a first flag.")
        if '--' in arg:
            arg_key = arg.lstrip('-')
            arg_dict[arg_key] = []
        else:
            arg_dict[arg_key].append(process(arg))

    for k, v in arg_dict.items():
        if len(v) == 0:
            v.append(True)

    given_shorthands = dict()
    fixed_keys = list(arg_dict.keys())
    for k in fixed_keys:
        p1, p2 = k.find('['), k.find(']')
        if p1 >= 0 and p2 >= 0:
            k_new = k[:p1]
            shorthand = k[p1+1:p2]
            given_shorthands[k_new] = shorthand
            arg_dict[k_new] = arg_dict[k]
            del arg_dict[k]

    for special_name, true_name in SUBSTITUTIONS.items():
        if special_name in arg_dict:
            arg_dict[true_name] = arg_dict[special_name]
            del arg_dict[special_name]

        if special_name in given_shorthands:
        given_shorthands[true_name] = given_shorthands[special_name]
        del given_shorthands[special_name]

    run_kwargs = dict()
    for k in RUN_KEYS:
        if k in arg_dict:
            val = arg_dict[k]
            assert len(val) == 1, \
                friendly_err("You can only provide one value for %s." % k)
            run_kwargs[k] = val[0]
            del arg_dict[k]

    if 'exp_name' in arg_dict:
        assert len(arg_dict['exp_name']) ==1, \
            friendly_err("You can only provide one value for exp_name.")
        exp_name = arg_dict['exp_name'][0]
        del arg_dict['exp_name'][0]
    else:
        exp_name = 'cmd_' + cmd

    if 'num_cpu' in run_kwargs and not (run_kwargs['num_cpu'] == 1):
        assert cmd in add_with_backends(MPI_COMPATIBLE_ALGOS), \
            friendly_err("This algorithm can't be run with num_cpu > 1.")

    valid_envs = [e.id for e in list(gym.envs.registry.all())]
    assert 'env_name' in arg_dict, \
        friendly_err("You did not give a value for --env_name! Add one and try again.")
    for env_name in arg_dict['env_name']:
        err_msg = dedent("""
            
            %s is not registered with Gym.
            
            Recommendations:
            
                * Check for a typo (did you include the version tag?)
                
                * View the complete list of valid Gym environments at
                
                    http://gym.openai.com/envs/
                    
            """ % env_name)
        assert env_name in valid_envs, err_msg

    eg = ExperimentGrid(name = exp_name)
    for k, v in arg_dict.items():
        eg.add(k, v, shorthand=given_shorthands.get(k))
    eg.run(algo, **run_kwargs)

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'help'
    valid_algos = add_with_backends(BASE_ALGO_NAMES)
    valid_utils = ['plot', 'test_policy']
    valid_help = ['--help', '-h', 'help']
    valid_cmds = valid_algos + valid_utils + valid_help
    assert cmd in valid_cmds, \
        "Select an algorithm or utility which is implemented in Spinning Up."

    if cmd in valid_help:
        str_valid_cmds = '\n\t' + '\n\t'.join(valid_algos + valid_utils)
        help_msg = dedent("""
            Experiment in Spinning Up from the command line with
            
            \tpython -m sc2ai.spinup.run CMD [ARGS...]
            
            where CMD is a valid command. Current valid commands are:
            """) + str_valid_cmds
        print(help_msg)

        subs_list = ['--' + k.ljust(10) + 'for'.ljust(10) + '--' + v \
                     for k, v in SUBSTITUTIONS.items()]
        str_valid_subs = '\n\t' + '\n\t'.join(subs_list)
        special_info = dedent("""
            FYI: When running an algorithm, any keyword argument to the
            algorithm function can be used as a flag, eg
            
            \tpython -m sc2ai.spinup.run ppo --env HalhCheetah-v2 --clip_ratio 0.1
            
            If you need a quick refresher on valid kwargs, get the docstring with
            
            \tpython -m sc2ai.spinup.run [alog] --help
            
            See the "Running Experiments" docs page for more details.
            
            Also: Some common but long flags can be substituted for shorter
            ones. Valid substitutions are:  
            """) + str_valid_subs
        print(special_info)
    elif cmd in valid_utils:
        runfile = osp.join(osp.abspath(osp.dirname(__file__)), 'utils', cmd + '.py')
        args = [sys.executable if sys.executable else 'python' , runfile] + sys.argv[2:]
        subprocess.check_call(args, env=os.environ)
    else:
        args = sys.argv[2:]
        parse_and_execute_grid_search(cmd, args)






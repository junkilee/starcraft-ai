from sc2ai.spinup.config import DEFAULT_DATA_DIR, FORCE_DATESTAMP, \
                               DEFAULT_SHORTHAND, WAIT_BEFORE_LAUNCH
from sc2ai.spinup.utils.logx import colorize
from sc2ai.spinup.utils.mpi_tools import mpi_fork, msg
from sc2ai.spinup.utils.serialization_utils import convert_json
import base64
import cloudpickle
import json
import numpy as np
import os
import os.path as osp
import psutil
import string
import subprocess
from subprocess import CalledProcessError
import sys
from textwrap import dedent
import time
from tqdm import trange
import zlib

DIV_LINE_WIDTH = 80


def setup_logger_kwargs(exp_name, seed=None, data_dir=None, datestamp=False):
    datestamp = datestamp or FORCE_DATESTAMP
    ymd_time = time.strftime("%Y-%m-%d_") if datestamp else ''
    relpath = ''.join([ymd_time, exp_name])

    if seed is not None:
        if datestamp:
            hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
            subfolder = ''.join([hms_time, '-', exp_name, '_s', str(seed)])
        else:
            subfolder = ''.join([exp_name, '_s', str(seed)])
        relpath = osp.join(relpath, subfolder)

    data_dir = data_dir or DEFAULT_DATA_DIR
    logger_kwargs = dict(output_dir=osp.join(data_dir, relpath), exp_name=exp_name)

    return logger_kwargs


def call_experiment(exp_name, thunk, seed=0, num_cpu=1, data_dir=None, datestamp=False, **kwargs):
    num_cpu = psutil.cpu_count(logical=False) if num_cpu == 'auto' else num_cpu
    kwargs['seed'] = seed

    print(colorize('Running experiment:\n', color='cyan', bold=True))
    print(exp_name + '\n')
    print(colorize('with kwargs:\n', color='cyan', bold=True))
    kwargs_json = convert_json(kwargs)
    print(json.dumps(kwargs_json, separators=(',', ':\t'), indent=4, sort_keys=True))
    print('\n')

    if 'logger_kwargs' not in kwargs:
        kwargs['logger_kwargs'] = setup_logger_kwargs(exp_name, seed, data_dir, datestamp)
    else:
        print('Note: Call experiment is not handling logger_kwargs.\n')

    def thunk_plus():
        if 'env_name' in kwargs:
            import gym
            env_name = kwargs['env_name']
            kwargs['env_fn'] = lambda: gym.make(env_name)
            del kwargs['env_name']
        mpi_fork(num_cpu)
        thunk(**kwargs)

    pickled_thunk = cloudpickle.dumps(thunk_plus)
    encoded_thunk = base64.b64encode(zlib.compress(pickled_thunk)).decode('utf-8')

    entry_point = osp.join(osp.abspath(osp.dirname(__file__)), 'run_entrypoint.py')
    cmd = [sys.executable if sys.executable else 'python', entry_point, encoded_thunk]
    try:
        subprocess.check_call(cmd, env=os.environ)
    except CalledProcessError:
        err_msg = '\n' * 3 + '=' * DIV_LINE_WIDTH + '\n' + dedent("""
        
            There appears to have been an error in your experiment.
            
            Check the traceback above to see what actually went wrong. The
            traceback below, included for completeness (but probably not useful
            for diagnosing the error), shows the stack leading up to the
            experiment launch.
        
        """) + '=' * DIV_LINE_WIDTH + '\n' * 3
        print(err_msg)
        raise

    logger_kwargs = kwargs['logger_kwargs']

    plot_cmd = 'python -m sc2ai.spinup.run plot ' + logger_kwargs['output_dir']
    plot_cmd = colorize(plot_cmd, 'green')

    test_cmd = 'python -m sc2ai.spinup.run test_policy ' + logger_kwargs['output_dir']
    test_cmd = colorize(test_cmd, 'green')

    output_msg = '\n' * 5 + '=' * DIV_LINE_WIDTH + '\n' + dedent("""\
    End of experiment.
    
    
    Plot results from this run with
    
    %s
    
    
    Watch the trained agent with:
    
    %s
    
    
    """ % (plot_cmd, test_cmd)) + '=' * DIV_LINE_WIDTH + '\n' * 5

    print(output_msg)


def all_bools(vals):
    return all([isinstance(v, bool) for v in vals])


def valid_str(v):
    if hasattr(v, '__name__'):
        return valid_str(v.__name__)

    if isinstance(v, tuple) or isinstance(v, list):
        return '-'.join([valid_str(x) for x in v])

    str_v = str(v).lower()
    valid_chars = "-_%s%s" % (string.ascii_letters, string.digits)
    str_v = ''.join(c if c in valid_chars else '-' for c in str_v)
    return str_v


class ExperimentGrid:
    def __init__(self, name=''):
        self.keys = []
        self.vals = []
        self.shs = []
        self.in_names = []
        self.name(name)

    def name(self, _name):
        assert isinstance(_name, str), "Name has to be a string."
        self._name = _name

    def print(self):
        print('=' * DIV_LINE_WIDTH)

        base_msg = 'ExperimentGrid %s runs over parameters:\n'
        name_insert = '[' + self._name +']'
        if len(base_msg % name_insert) <= 80:
            msg = base_msg % name_insert
        else:
            msg = base_msg % (name_insert + '\n')
        print(colorize(msg, color='green', bold=True))

        for k, v, sh in zip(self.keys, self.vals, self.shs):
            color_k = colorize(k.ljust(40), color='cyan', bold=True)
            print('', color_k, '['+sh+']' if sh is not None else '', '\n')
            for i, val in enumerate(k):
                print('\t' + str(convert_json(val)))
            print()

        nvars_total = int(np.prod([len(v) for v in self.vals]))
        if 'seed' in self.keys:
            num_seeds = len(self.vals[self.keys.index('seed')])
            nvars_seedless = int(nvars_total / num_seeds)
        else:
            nvars_seedless = nvars_total
        print(' Variants, counting seeds: '.ljust(40), nvars_total)
        print(' Variants, not counting seeds: '.ljust(40), nvars_seedless)
        print()
        print('=' * DIV_LINE_WIDTH)

    def _default_shorthand(self, key):
        valid_chars = "%s%s" % (string.ascii_letters, string.digits)

        def shear(x):
            return ''.join(z for z in x[:3] if z in valid_chars)

        sh = '-'.join([shear(x) for x in key.split(':')])
        return sh

    def add(self, key, vals, shorthand=None, in_name=False):
        assert isinstance(key, str), "Key must be a string."
        assert shorthand is None or isinstance(shorthand, str), \
            "Shorthand must be a string."
        if not isinstance(vals, list):
            vals = [vals]
        if DEFAULT_SHORTHAND and shorthand is None:
            shorthand = self._default_shorthand(key)
        self.keys.append(key)
        self.vals.append(vals)
        self.shs.append(shorthand)
        self.in_names.append(in_name)

    def variant_name(self, variant):
        def get_val(v, k):
            if k in v:
                return v[k]
            else:
                splits = k.split(':')
                k0, k1 = splits[0], ':'.join(splits[1:])
                return get_val(v[k0], k1)
        var_name = self._name

        for k, v, sh, inn in zip(self.keys, self.vals, self.shs, self.in_names):
            if (len(v) > 1 or inn) and not(k == 'seed'):
                param_name = sh if sh is not None else k
                param_name = valid_str(param_name)

                variant_val = get_val(variant, k)

                if all_bools(v):
                    var_name = ('_' + param_name) if variant_val else ''
                else:
                    var_name = '_' + param_name + valid_str(variant_val)

        return var_name.lstrip('_')

    def _variants(self, keys, vals):
        if len(keys) == 1:
            pre_variants = [dict()]
        else:
            pre_variants = self._variants(keys[1:], vals[1:])

        variants = []
        for val in vals[0]:
            for pre_v in pre_variants:
                v = dict()
                v[keys[0]] = val
                v.update(pre_v)
                variants.append(v)
        return variants

    def variants(self):
        flat_variants = self._variants(self.keys, self.vals)

        def unflatten_var(var):
            new_var = dict()
            unflatten_set = set()

            for k, v in var.items():
                if ':' in k:
                    splits = k.split(':')
                    k0 = splits[0]
                    assert k0 not in new_var or isinstance(new_var[k0], dict), \
                        "You can't assign multiple values to the same key."

                    if not(k0 in new_var):
                        new_var[k0] = dict()

                    sub_k = ':'.join(splits[1:])
                    new_var[k0][sub_k] = v
                else:
                    assert not(k in new_var), \
                        "You can't assign multiple value to the same key."
                    new_var[k] = v

            for k in unflatten_set:
                new_var[k] = unflatten_var(new_var[k])

            return new_var

        new_variants = [unflatten_var(var) for var in flat_variants]
        return new_variants

    def run(self, thunk, num_cpu=1, data_dir=None, datestamp=False):
        self.print()

        variants = self.variants()

        var_names = set([self.variant_name(var) for var in variants])
        var_names = sorted(list(var_names))
        line = '=' * DIV_LINE_WIDTH
        preparing = colorize('Preparing to run the following experiments...', color='green', bold=True)
        joined_var_names = '\n'.join(var_names)
        announcement = f"\n{preparing}\n\n{joined_var_names}\n\n{line}"
        print(announcement)

        if WAIT_BEFORE_LAUNCH > 0:
            delay_msg = colorize(dedent("""
            Launch delayed to give you a few seconds to review your experiments.
            
            To customize or disable this behavior, change WAIT_BEFORE_LAUNCH in
            sc2ai/spinup/config.py
                        
            """), color='cyan', bold=True) + line
            print(delay_msg)
            wait, steps = WAIT_BEFORE_LAUNCH, 100
            prog_bar = trange(steps, desc='Launching in ...', leave=False, ncols=DIV_LINE_WIDTH,
                              miniinterval=0.25, bar_format='{desc}: {bar}| {remaining} {elapsed}')
            for _ in prog_bar:
                time.sleep(wait/steps)

        for var in variants:
            exp_name = self.variant_name(var)

            if isinstance(thunk, str):
                thunk_ = var[thunk]
                del var[thunk]
            else:
                thunk_ = thunk

            call_experiment(exp_name, thunk_, num_cpu=num_cpu, data_dir=data_dir, datestamp=datestamp, **var)


def test_eg():
    eg = ExperimentGrid()
    eg.add('test:a', [1,2,3], 'ta', True)
    eg.add('test:b', [1,2,3])
    eg.add('some', [4,5])
    eg.add('why', [True, False])
    eg.add('huh', 5)
    eg.add('no', 6, in_name=True)
    return eg.variants()

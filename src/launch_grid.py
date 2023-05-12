import fcntl
import itertools
import json
import os
import pickle
import random
from argparse import Namespace
from datetime import datetime
from os.path import exists as file_exists
from time import sleep
from pathlib import Path
import wandb
from train_sbatch_grid import get_arg_parser, run_training_process_with_validation
from train_regression import run_training_process_with_validation as run_regression

# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["WANDB_SILENT"] = "true"


def get_dict(to_dict):
    return to_dict if isinstance(to_dict, dict) else to_dict.__dict__


def get_grid(n=0, idx=0):
    config = {'group': None, 'project': None, 'grid': None}
    filename = f'config_{n}.json'
    if not Path(filename).is_file():
        print(f'{filename} not exist. Exiting')
        raise FileNotFoundError()
    with open(filename) as f:
        config = json.load(f)
    is_list_file = isinstance(config, list)
    if is_list_file:
        if idx >= len(config):
            idx = -1
        else:
            config = config[idx]
    return config['group'], config['project'], config['grid'], is_list_file, idx


def is_param_list_in_wandb(params, runs_conf, keys_to_check=None):
    if keys_to_check is None:
        keys_to_check = params.keys()
    for r in runs_conf:
        same = True
        for k in keys_to_check:
            if r.get(k) is None or params.get(k) is None:
                continue
            if isinstance(params[k], list):
                for i in range(len(params[k])):
                    same = same and params[k][i] == r[k][i]
            else:
                same = same and params[k] == r[k]
        if same:
            return r
    return False


def wandb_checker(all_confs, group, project, runs_count=0):
    test = True

    try:
        api = wandb.Api(timeout=60)
        runs = api.runs(f"{group}/{project}")
        runs_conf = [crun.config for crun in runs if crun.state not in ['crashed', 'failed']
                     or (crun.state == 'finished' and crun.summary.get('test_acc') is not None)]
        [crun.delete() for crun in runs if crun.state in ['crashed', 'failed'] or (
            crun.state == 'finished' and crun.summary.get('test_acc') is None)]
        # for crun in runs:
        #     print(crun.id)
        #     history = crun.history()
        #     if crun.state == 'running' and history.empty:
        #         print(crun.id)
        #         continue
        #     if crun.state == 'running' and len(history.loc[history['epoch'] >= 100]) > 0 and history.loc[history['epoch'] >= 100].iloc[0]['acc'] < 60:
        #         # if the run is still running and the accuracy is less than 60% stop the run
        #         crun.finish(999, True)
        #     elif crun.state in ['finished', 'crashed'] and (crun.summary.get('test_acc') is not None and crun.summary['test_acc'] < 60 or crun.summary.get('acc') is not None and crun.summary['acc'] < 60):
        #         crun.delete()
        #     if crun.state not in ['crashed', 'failed']:
        #         runs_conf.append(crun.config)
        print(f"FOUND {len(runs_conf)} RUNS ON WANDB")
        print(f"Checking {len(all_confs)} CONFIGURATIONS")
        grid_keys = get_grid().keys()
        params_list = [p for p in all_confs if not is_param_list_in_wandb(get_dict(p), runs_conf, grid_keys)]
        if len(params_list) == 0:
            raise NameError()
    except Exception as ex:
        print(ex)
        if runs_count < 10:
            sleep(10)
            return wandb_checker(all_confs, group, project, runs_count + 1)
        test = False
    return params_list if test else []


def local_wandb_checker(all_confs, group, project, grid):
    data = None
    runs_conf = None
    conf = None
    file = f'./configs_{project}.pickle'
    try:
        f = open(file, 'rb+')
        data = pickle.load(f)
    except FileNotFoundError as ex:
        f = open(file, 'wb')

    fcntl.flock(f, fcntl.LOCK_EX)
    print('Unlocked')
    try:
        api = wandb.Api(timeout=60)
        if data is None:
            print('New data')
            runs = api.runs(f"{group}/{project}", {"$and": [{"state": {"$nin": ["crashed", "failed"]}}]})
            print(f'Found {len(runs)} clean runs')
            runs_conf = [{'id': run.id, **run.config} for run in runs if run.config.get('dataset') is not None]
        else:
            print('Read data')
            runs_conf = data
            ids = [run['id'] for run in data if run.get('id')]
            print(f'Found {len(ids)} saved runs')
            runs = api.runs(f"{group}/{project}",
                            {"$and": [{"state": {"$nin": ["crashed", "failed"]}}, {"name": {"$nin": ids}}]})
            filter_in_runs = [{'id': run.id, **run.config} for run in runs if run.config.get('dataset') is not None]
            print(f'Found {len(filter_in_runs)} clean runs')
            runs = api.runs(f"{group}/{project}", {"$and": [{"state": {"$in": ["crashed", "failed"]}}]})
            filter_out_runs = [run.id for run in runs]
            print(f'Found {len(filter_out_runs)} wasted runs')
            runs_conf = [run for run in runs_conf if run.get('in_progress') or (run.get('id') is not None and
                         run['id'] not in filter_out_runs and run.get('dataset') is not None)] + filter_in_runs

        print(f'Found {len(runs_conf)} runs on wandb')
        print(f'Checking {len(all_confs)} configurations')
        for c in all_confs:
            if not is_param_list_in_wandb(get_dict(c), runs_conf, grid.keys()):
                c['fake_id'] = random.randint(0, 10 ** 15)
                c['in_progress'] = 1
                conf = c
                runs_conf.append(conf)
                print(f'''Found new configuration. Id: {c['fake_id']}''')
                break
        f.seek(0)
        pickle.dump(runs_conf, f)
    except Exception as ex:
        raise ex
    finally:
        f.close()

    return conf


if __name__ == "__main__":
    print('hello', datetime.now())
    params = get_arg_parser().parse_args()
    n = 0
    conf = None
    group = ''
    project = ''
    new_group = ''
    new_project = ''
    idx_config = None
    while conf is None and file_exists(f'config_{n}.json'):
        if idx_config is None:
            new_group, new_project, grid, is_list_file, idx_config = get_grid(n)
        else:
            new_group, new_project, grid, is_list_file, idx_config = get_grid(n, idx_config)
        entered = True
        while conf is None:
            if entered:
                entered = False
            else:
                new_group, new_project, grid, is_list_file, idx_config = get_grid(n, idx_config)
            if idx_config < 0:
                break
            group, project = new_group, new_project
            print(group, project)
            all_confs = [{k: val for k, val in zip(grid, v)} for v in itertools.product(*[grid[k] for k in grid])]
            random.shuffle(all_confs)
            conf = local_wandb_checker(all_confs, group, project, grid)
            if not is_list_file:
                break
            idx_config += 1
        n += 1
    if conf is not None:
        print(f'Dataset: {conf["dataset"]}')
        conf['group'] = group
        conf['project'] = project
        conf['filename'] = params.filename
        default = get_arg_parser().parse_args([]).__dict__
        for k in conf.keys():
            default[k] = conf[k]
        default['date'] = f'{datetime.utcnow():%Y-%m-%d %H:%MZ}'
        try:
            if default['dataset'] == 'ZINC':
                run_regression(Namespace(**default))
            else:
                run_training_process_with_validation(Namespace(**default))
        except Exception as ex:
            raise ex
        finally:
            if conf.get('fake_id') is not None:
                file = f'./configs_{project}.pickle'
                with open(f'./configs_{project}.pickle', 'rb+') as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    data = pickle.load(f)
                    f.seek(0)
                    pickle.dump([run for run in data if run.get('fake_id') != conf['fake_id']], f)

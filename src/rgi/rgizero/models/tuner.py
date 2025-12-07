import time
import json
import os
import ast
from typing import Any, Callable
from pprint import pprint

def rewrite_cache_file(path, defaults):
    """Rewrite the cache file to match the current defaults."""
    data=json.load(open(path))
    new_data={}; 
    for k,v in data.items():
        if k == 'best_model_trajectory':
            new_data[k] = v
            continue
        
        d = dict(eval(k)); 
        for dk, dv in defaults.items():
            d.setdefault(dk, dv)
        new_data[str(sorted(d.items()))] = v
    json.dump(new_data, open(path, 'w'))


def clear_failures_from_cache_file(path, max_sane_val=1_000_000):
    data=json.load(open(path))
    new_data={}; 
    for k,v in data.items():
        if k == 'best_model_trajectory' or v['val'] < max_sane_val:
            new_data[k] = v
    json.dump(new_data, open(path, 'w'))


import dataclasses
from rgi.rgizero.models.transformer import TransformerConfig
from rgi.rgizero.train import TrainConfig
import torch
from rgi.rgizero.models.action_history_transformer import ActionHistoryTransformer
import numpy as np

transform_config_fields = {f.name for f in dataclasses.fields(TransformerConfig)}
train_config_fields = {f.name for f in dataclasses.fields(TrainConfig)}

print(f'transform_config_fields: {transform_config_fields}')
print(f'train_config_fields: {train_config_fields}')

def create_random_model(config: TransformerConfig, action_vocab_size, num_players,  seed: int, device: str):
    torch.manual_seed(seed)
    np.random.seed(seed) # Ensure numpy operations are also seeded
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    model = ActionHistoryTransformer(config=config, action_vocab_size=action_vocab_size, num_players=num_players)
    model.to(device)
    return model

from rgi.rgizero.data.trajectory_dataset import build_trajectory_loader
from rgi.rgizero.train import Trainer

def train_model(model, training_splits, train_config, device: str, n_max_context: int, data_dir: str, num_workers: int = 0):
    # Load dataset
    trajectory_loader = build_trajectory_loader(
        data_dir, training_splits, block_size=n_max_context, batch_size=train_config.batch_size,
        device=device, workers=num_workers, shuffle=True)
        
    trainer = Trainer(
        model=model,
        train_config=train_config,
        train_loader=trajectory_loader,
        val_loader=trajectory_loader,  # TODO: Create separate validation loader
        device=device
    )

    trainer.train()
    return model, trainer

def train_with(vocab_size, num_players, num_genrations, device, n_max_context, data_dir,**overrides):
    """Wrapper fn to train a model using the latest train.py code and the given overrides."""
    t0 = time.time()

    for override in overrides:
        if override not in transform_config_fields and override not in train_config_fields:
            raise ValueError(f"Invalid override: {override}")


    model_config_overrides = {k:v for k,v in overrides.items() if k in transform_config_fields}
    train_config_overrides = {k:v for k,v in overrides.items() if k in train_config_fields}

    model_config = TransformerConfig(**model_config_overrides)
    train_config = TrainConfig(**train_config_overrides)

    print(f"model_config={model_config}")
    print(f"train_config={train_config}")
    model = create_random_model(model_config, action_vocab_size=vocab_size, num_players=num_players, seed=42, device=device)

    training_splits = [f'gen-{generation_id}' for generation_id in range(1, num_genrations+1)]

    model, trainer = train_model(model, training_splits, train_config, device=device, n_max_context=n_max_context, data_dir=data_dir)
    loss_dict = trainer.estimate_loss()
    loss_dict = {k: float(v) for k, v in loss_dict.items()}

    # def train_model(model, training_splits, train_config):
    # loss_dict = train.train_and_evaluate(**overrides)
    elapsed = time.time() - t0
    print(f"## train_loss: {loss_dict['train']:.4f}, val_loss: {loss_dict['val']:.4f}, Time taken: {elapsed}s, overrides={overrides}")
    return loss_dict, elapsed


class Tuner:
    """Class to automate the choice of model hyperparameters to reduce loss."""
    def __init__(self, 
        fixed_params: dict[str, Any],
        initial_params: dict[str, Any],
        tune_options: dict[str, list[Any]],
        computed_tune_options: dict[str, Callable[[dict[str, Any]], list[Any]]],
        cache_version: str,
        target_improvement_per_minute: float = 0.0,
        initialize_from_best_model: bool = True,
        save_trained_models: bool = False,
    ):
        """
        args:
            tune_options: dict[str, list[Any]]
                Dict from hpyerparameter name to sorted list of possible values.
            initial_params: dict[str, Any]
                Dict from hyperparameter name to initial value.
            cache_version: str
                A version string for the cache file name.
        """
        self.fixed_params = fixed_params.copy()        
        self.tune_options = tune_options.copy()
        self.computed_tune_options = computed_tune_options.copy()

        initial_params_keys = set(initial_params.keys())
        fixed_keys = set(fixed_params.keys())
        tune_keys = set(tune_options.keys())
        computed_tune_keys = set(computed_tune_options.keys())
        all_tune_keys = tune_keys | computed_tune_keys
        
        if tune_keys & computed_tune_keys:
            raise ValueError(f"Duplicate keys found in tune_keys and comuted_tune_keys -> {tune_keys & computed_tune_keys}")
        if fixed_keys & all_tune_keys:
            raise ValueError(f"Can't tune fixed keys -> {fixed_keys & all_tune_keys}")
        if fixed_keys & initial_params_keys:
            raise ValueError(f"Duplicate fixed and initial keys -> {fixed_keys & initial_params_keys}")
        if all_tune_keys != initial_params_keys:
            raise ValueError(f"all_tune_keys != initial_params_keys. Added: {initial_params_keys - all_tune_keys}. Removed: {all_tune_keys - initial_params_keys}")

        self.target_improvement_per_minute = target_improvement_per_minute
        self.target_improvement_per_second = target_improvement_per_minute / 60
        self.save_trained_models = save_trained_models

        # Load cache file.
        self.cache_path = f'result_cache-v{cache_version}.json'
        self.result_cache = json.load(open(self.cache_path)) if os.path.exists(self.cache_path) else {}

        self.best_hparams_path = f'best_hparams-v{cache_version}.json'
        self.best_model_trajectory = []

        if initialize_from_best_model and self.result_cache:
            best_cache_entry = min(self.result_cache.items(), key=lambda kv: kv[1]['val'] + kv[1]['elapsed'] * self.target_improvement_per_second)
            best_cache_params = dict(ast.literal_eval(best_cache_entry[0]))
            initial_params = best_cache_params
            new_initial_params_keys = initial_params.keys()
            if initial_params_keys != new_initial_params_keys:
                raise ValueError(f"initial_params_keys != new_initial_params_keys after loadding best params. Added: {new_initial_params_keys - initial_params_keys}. Removed: {initial_params_keys - new_initial_params_keys}")


        self.current_params = initial_params.copy()
        self.current_params.update(self.fixed_params)

        # Create initial_param set to begin tuning from.
        # default_params = train.Hyperparameters()
        # self.initial_params = {k: (initial_params[k] if k in initial_params else getattr(default_params, k)) for k in tune_options.keys()}
        # self.initial_params = {k: initial_params[k] for k in tune_options.keys()}
        # for arg, val in self.initial_params.items():
        #     if val not in tune_options[arg]:
        #         raise Exception(f"Value {arg}={val} not in {tune_options[arg]}")

        self.best_params = None
        self.best_loss = None
        self.generation = 0

    def _save_result_cache(self):
        json.dump(self.result_cache, open(self.cache_path, 'w'))

    def train_and_compute_loss(self, params) -> tuple[float, float]:
        """Look up loss in cache, or train model to compute it and save to cache."""
        param_key = str(sorted(params.items()))
        if param_key in self.result_cache:
            loss_dict = self.result_cache[param_key]
        else:
            try:
                loss_dict, elapsed = train_with(**params)
                loss_dict['elapsed'] = elapsed
            except Exception as e:
                import traceback
                print(f"Error training with params {params}: error='{e}' traceback='{traceback.format_exc()}'")
                loss_dict = {'val': float('inf'), 'elapsed': float('inf')}
            self.result_cache[param_key] = loss_dict
            self._save_result_cache()
        return loss_dict['val'], loss_dict['elapsed'], loss_dict
    
    def maybe_update_best_param(self, loss, elapsed, params, loss_dict):
        """If model is improved, add it to result_cache['best_model_trajectory']."""
        if self.best_loss is None:
            self.best_loss = loss
            self.best_loss_elapsed = elapsed
            self.best_params = params.copy()
            self.result_cache['best_model_trajectory'].append({
                'change': [],
                'loss': loss,
                'loss_delta': 0,
                'elapsed': elapsed,
                'args': params.copy(),
                'loss_dict': loss_dict.copy(),
            })
            return True
        
        # lower loss is better, lower elapsed is better, so lower score is better.
        best_loss_score = self.best_loss + self.best_loss_elapsed * self.target_improvement_per_second
        current_loss_score = loss + elapsed * self.target_improvement_per_second
        if current_loss_score >= best_loss_score:
            return False

        prev_best_loss = self.best_loss
        prev_best_loss_elapsed = self.best_loss_elapsed
        prev_best_params = self.best_params
        self.best_loss = loss
        self.best_loss_elapsed = elapsed
        self.best_params = params.copy()

        all_keys = sorted(set(self.best_params.keys()) | set(prev_best_params.keys()))
        changed_params = [{'k':k, 'old': prev_best_params.get(k), 'new':self.best_params.get(k)} for k in all_keys if prev_best_params.get(k) != self.best_params.get(k)]

        self.result_cache['best_model_trajectory'].append({
            'change': changed_params,
            'loss': loss,
            'loss_delta': prev_best_loss - loss,            
            'elapsed': elapsed,
            'elapsed_delta': prev_best_loss_elapsed - elapsed,
            'args': params.copy(),
            'loss_dict': loss_dict.copy(),
        })
        self._save_result_cache()
        return True

    def autotune(self, num_generations=10) -> bool:
        """Autotune the model by training and evaluating it with different hyperparameters.
        
        Algorithm:
        - Train initial model.
        - For each generation:
            - For each hyperparameter:
                - Tune the hyperparameter by increasing it until loss stops improving.
                - If no improvement was made, decrease the hyperparameter until loss stops improving.
                - If any improvment was made, update the default parameters and move on to tuning the next hyperparameter.
        """
        if self.best_loss is None:
            loss, elapsed, loss_dict = self.train_and_compute_loss(self.initial_params)
            self.maybe_update_best_param(loss, elapsed, self.initial_params, loss_dict)
            print(f"## Initial Model, loss={self.best_loss} elapsed={self.best_loss_elapsed}s")

        new_best_model_found = False
        for generation in range(1, num_generations+1):
            self.generation = generation
            new_best_model_found_this_generation = False
            for param_name in self.tune_options.keys():
                prev_best_loss = self.best_loss
                prev_best_loss_elapsed = self.best_loss_elapsed
                print(f"## Tuning generation {self.generation}: {param_name}")
                is_improved = self.tune_hyperparameter(param_name)
                if is_improved:
                    print(f"## Tuning generation {self.generation}: {param_name} improved, val={self.best_params[param_name]}, best={self.best_loss}, delta={prev_best_loss - self.best_loss} elapsed={self.best_loss_elapsed}s delta={prev_best_loss_elapsed - self.best_loss_elapsed}s")
                new_best_model_found_this_generation |= is_improved
            if not new_best_model_found_this_generation:
                print(f"No updates for generation {generation}, stopping")
                return new_best_model_found
            new_best_model_found = True


    def _tune_hyperparameter_range(self, params, param_name, idx_list) -> bool:
        """Try each idx in idx_list for param. Return as soon as one fails to update best_params. Return True if any updates succeed."""
        best_updated = False

        for idx in idx_list:
            params[param_name] = self.tune_options[param_name][idx]
            loss, elapsed, loss_dict = self.train_and_compute_loss(params)
            if self.maybe_update_best_param(loss, elapsed, params, loss_dict):
                best_updated = True
            else:
                return best_updated
        return best_updated            

    def tune_hyperparameter(self, param_name) -> bool:
        params = self.best_params.copy()
        if param_name in self.computed_tune_options:
            self.tune_options[param_name] = self.computed_tune_options[param_name](params)
            print(f"## Computed tune options: {param_name} = {self.tune_options[param_name]}")
            if params[param_name] not in self.tune_options[param_name]:
                if self.tune_options[param_name] != sorted(self.tune_options[param_name]):
                    raise Exception(f"Computed tune options {param_name} = {self.tune_options[param_name]} not sorted")
                # Use the value one-below the target value if target not found.
                for i in range(len(self.tune_options[param_name])):
                    if self.tune_options[param_name][i] > params[param_name]:
                        break                
                params[param_name] = self.tune_options[param_name][max(0,i-1)]
                
        current_idx = self.tune_options[param_name].index(params[param_name])

        # Try tuning from current_idx updwrds.
        if self._tune_hyperparameter_range(params, param_name, range(current_idx+1, len(self.tune_options[param_name]))):
            return True
        # Try tuning from current_idx downwards.
        if self._tune_hyperparameter_range(params, param_name, range(current_idx-1, -1, -1)):
            return True
        # No updates improved best_loss
        return False

    def print_hparam_stats(self):
        import ast
        from collections import defaultdict
        
        # for str_key, val in self.result_cache.items():
        #     if str_key == 'best_model_trajectory':
        #         continue
        #     list_key = ast.literal_eval(str_key)
        #     print(f"## key={list_key}")
        #     print(f"## val={val}")
        # # return None

        # tree[param][remaining-key][param-val] = loss_dict
        tree = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
        for str_key, eval_dict in self.result_cache.items():
            if str_key == 'best_model_trajectory':
                continue
            list_key = ast.literal_eval(str_key)
            for param_idx, (param_name, param_val) in enumerate(list_key):
                remaining_key = str(list_key[:param_idx]+list_key[param_idx+1:])
                tree[param_name][remaining_key][param_val] = eval_dict
        # print(f"## tree={tree}")
        # [d for v in tree['max_iters'].values() for d in [dict(v)] if len(d) > 1][0]

        # grouped_loss_dicts[param_name][val1][val2] = list_of_pairs
        grouped_loss_dicts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))
        for param_name, xxx in tree.items():
            for remaining_key, xx in xxx.items():
                for param_val_1, loss_dict_1 in xx.items():
                    for param_val_2, loss_dict_2 in xx.items():
                        if param_val_1 != param_val_2:
                            grouped_loss_dicts[param_name][param_val_1][param_val_2].append((loss_dict_1, loss_dict_2))
        # print(f"## deltas={grouped_loss_dicts}")

        stats_dict = {}
        for param_name, v1_v2_loss_dicts in grouped_loss_dicts.items():
            for val1, v2_loss_dicts in v1_v2_loss_dicts.items():
                for val2, loss_dicts in v2_loss_dicts.items():
                    #print(f"## {param_name} {val1} {val2} {loss_dicts}")
                    stats_dict[(param_name, val1, val2)] = self._compute_stats(loss_dicts)
        return stats_dict

    def _compute_stats(self, loss_dicts):
        import numpy as np
        stats_dict = {
            'mean_val_1': np.mean([loss_dict[0]['val'] for loss_dict in loss_dicts]),
            'mean_val_2': np.mean([loss_dict[1]['val'] for loss_dict in loss_dicts]),
            'mean_val_delta': np.mean([loss_dict[0]['val'] - loss_dict[1]['val'] for loss_dict in loss_dicts]),

            'mean_elapsed_1': np.mean([loss_dict[0]['elapsed'] for loss_dict in loss_dicts]),
            'mean_elapsed_2': np.mean([loss_dict[1]['elapsed'] for loss_dict in loss_dicts]),
            'mean_elapsed_delta': np.mean([loss_dict[0]['elapsed'] - loss_dict[1]['elapsed'] for loss_dict in loss_dicts]),

            'std_val_1': np.std([loss_dict[0]['val'] for loss_dict in loss_dicts]),
            'std_val_2': np.std([loss_dict[1]['val'] for loss_dict in loss_dicts]),

            'std_elapsed_1': np.std([loss_dict[0]['elapsed'] for loss_dict in loss_dicts]),
            'std_elapsed_2': np.std([loss_dict[1]['elapsed'] for loss_dict in loss_dicts]),
        }
        # Assume normal distribution, Calculate how many standard deviations away from the mean the delta is.
        stats_dict['std_val_delta'] = stats_dict['mean_val_delta'] / stats_dict['mean_val_1']
        stats_dict['std_elapsed_delta'] = stats_dict['mean_elapsed_delta'] / stats_dict['mean_elapsed_1']

        stats_dict = {k: float(v) for k, v in stats_dict.items()}
        return stats_dict

def retune_model(initial_params, tune_options, param_overrides, tune_option_overrides, cache_version, num_generations=10, out_label=None):
    # Tune single epoch model.
    initial_params = initial_params.copy()
    tune_options = tune_options.copy()

    initial_params.update(param_overrides)
    tune_options.update(tune_option_overrides)

    tuner = Tuner(tune_options, initial_params, cache_version=cache_version)
    tuner.autotune(num_generations=num_generations)

    best_model_result = tuner.result_cache['best_model_trajectory'][-1]
    best_model_params = best_model_result['args']

    if out_label:
        with open(f'out_{out_label}.json', 'w') as f:
            json.dump(tuner.result_cache['best_model_trajectory'], f)

    print('\n## Best model:')
    pprint(best_model_result)
    return tuner, best_model_params
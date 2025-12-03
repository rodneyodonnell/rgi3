import time
import json
import os
from typing import Any

def rewrite_cache_file(path, defaults):
    """Rewrite the cache file to match the current defaults."""
    data=json.load(open(path))
    new_data={}; 
    for k,v in data.items():
        if k == 'best_model_trajectory':
            new_data[k] = v;
            continue;
        
        d = dict(eval(k)); 
        for dk, dv in defaults.items():
            d.setdefault(dk, dv);
        new_data[str(sorted(d.items()))] = v
    json.dump(new_data, open(path, 'w'))

def train_with(**overrides):
    """Wrapper fn to train a model using the latest train.py code and the given overrides."""
    t0 = time.time()
    loss_dict = train.train_and_evaluate(**overrides)
    elapsed = time.time() - t0
    print(f"## corrected_loss: {loss_dict['corrected_loss']:.4f}, original_loss: {loss_dict['val_loss']:.4f}, Time taken: {elapsed}s, overrides={overrides}")
    return loss_dict, elapsed

class Tuner:
    """Class to automate the choice of model hyperparameters to reduce loss."""
    def __init__(self, tune_options: dict[str, list[Any]], initial_params: dict[str, Any], cache_version: str):
        """
        args:
            tune_options: dict[str, list[Any]]
                Dict from hpyerparameter name to sorted list of possible values.
            initial_params: dict[str, Any]
                Dict from hyperparameter name to initial value.
            cache_version: str
                A version string for the cache file name.
        """
        self.tune_options = tune_options

        # Load cache file.
        self.cache_path = f'result_cache-v{cache_version}.json'
        self.result_cache = json.load(open(self.cache_path)) if os.path.exists(self.cache_path) else {}
        # Clear trajectory. We'll rebuild this in autotune()
        self.result_cache['best_model_trajectory'] = []

        # Create initial_param set to begin tuning from.
        default_params = train.Hyperparameters()
        self.initial_params = {k: (initial_params[k] if k in initial_params else getattr(default_params, k)) for k in tune_options.keys()}
        for arg, val in self.initial_params.items():
            if val not in tune_options[arg]:
                raise Exception(f"Value {arg}={val} not in {tune_options[arg]}")

        self.best_params = None
        self.best_loss = None
        self.generation = 0

        print(f"Initial params: {self.best_params}")

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
                print(f"Error training with params {params}: {e}")
                loss_dict = {'corrected_loss': float('inf'), 'elapsed': float('inf')}
            self.result_cache[param_key] = loss_dict
            self._save_result_cache()
        return loss_dict['corrected_loss'], loss_dict['elapsed'], loss_dict
    
    def maybe_update_best_param(self, loss, elapsed, params, loss_dict):
        """If model is improved, add it to result_cache['best_model_trajectory']."""
        if self.best_loss is None:
            self.best_loss = loss
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
        
        if loss >= self.best_loss:
            return False

        prev_best_loss = self.best_loss
        prev_best_params = self.best_params
        self.best_loss = loss
        self.best_params = params.copy()

        all_keys = sorted(set(self.best_params.keys()) | set(prev_best_params.keys()))
        changed_params = [{'k':k, 'old': prev_best_params.get(k), 'new':self.best_params.get(k)} for k in all_keys if prev_best_params.get(k) != self.best_params.get(k)]

        self.result_cache['best_model_trajectory'].append({
            'change': changed_params,
            'loss': loss,
            'loss_delta': prev_best_loss - loss,
            'elapsed': elapsed,
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
            print(f"## Initial Model, loss={self.best_loss}")

        new_best_model_found = False
        for generation in range(1, num_generations+1):
            self.generation = generation
            new_best_model_found_this_generation = False
            for param_name in self.tune_options.keys():
                prev_best_loss = self.best_loss
                print(f"## Tuning generation {self.generation}: {param_name}")
                is_improved = self.tune_hyperparameter(param_name)
                if is_improved:
                    print(f"## Tuning generation {self.generation}: {param_name} improved, val={self.best_params[param_name]}, best={self.best_loss}, delta={prev_best_loss - self.best_loss}")
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
        current_idx = self.tune_options[param_name].index(params[param_name])

        # Try tuning from current_idx updwrds.
        if self._tune_hyperparameter_range(params, param_name, range(current_idx+1, len(self.tune_options[param_name]))):
            return True
        # Try tuning from current_idx downwards.
        if self._tune_hyperparameter_range(params, param_name, range(current_idx-1, -1, -1)):
            return True
        # No updates improved best_loss
        return False


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
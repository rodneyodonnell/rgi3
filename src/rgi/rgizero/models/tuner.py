import ast
import dataclasses
import hashlib
import json
import os
import time
from collections import defaultdict
from pprint import pprint
from typing import Any, Callable

import numpy as np
import torch

from rgi.rgizero.data.trajectory_dataset import build_trajectory_loader
from rgi.rgizero.models.action_history_transformer import ActionHistoryTransformer
from rgi.rgizero.models.transformer import TransformerConfig
from rgi.rgizero.train import TrainConfig, Trainer


def rewrite_cache_file(path, defaults):
    """Rewrite the cache file to match the current defaults."""
    data = json.load(open(path))
    new_data = {}
    for k, v in data.items():
        if k == "best_model_trajectory":
            new_data[k] = v
            continue

        d = dict(eval(k))
        for dk, dv in defaults.items():
            d.setdefault(dk, dv)
        new_data[str(sorted(d.items()))] = v
    json.dump(new_data, open(path, "w"))


def clear_failures_from_cache_file(path, max_sane_val=1_000_000):
    data = json.load(open(path))
    new_data = {}
    for k, v in data.items():
        if k == "best_model_trajectory" or v["val"] < max_sane_val:
            new_data[k] = v
    json.dump(new_data, open(path, "w"))


transform_config_fields = {f.name for f in dataclasses.fields(TransformerConfig)}
train_config_fields = {f.name for f in dataclasses.fields(TrainConfig)}

print(f"transform_config_fields: {transform_config_fields}")
print(f"train_config_fields: {train_config_fields}")


def create_random_model(config: TransformerConfig, action_vocab_size, num_players, seed: int, device: str):
    torch.manual_seed(seed)
    np.random.seed(seed)  # Ensure numpy operations are also seeded
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    model = ActionHistoryTransformer(config=config, action_vocab_size=action_vocab_size, num_players=num_players)
    model.to(device)
    return model


def train_model(model, dataset_paths, train_config, device: str, n_max_context: int, num_workers: int = 0):
    # Load dataset
    train_loader, val_loader = build_trajectory_loader(
        dataset_paths,
        block_size=n_max_context,
        batch_size=train_config.batch_size,
        device=device,
        workers=num_workers,
        shuffle=True,
    )

    trainer = Trainer(
        model=model, train_config=train_config, train_loader=train_loader, val_loader=val_loader, device=device
    )

    trainer.train()
    return model, trainer


def train_with(vocab_size, num_players, device, n_max_context, dataset_paths, **overrides):
    """Wrapper fn to train a model using the latest train.py code and the given overrides."""
    t0 = time.time()

    for override in overrides:
        if override not in transform_config_fields and override not in train_config_fields:
            raise ValueError(f"Invalid override: {override}")

    model_config_overrides = {k: v for k, v in overrides.items() if k in transform_config_fields}
    model_config_overrides["n_max_context"] = n_max_context
    train_config_overrides = {k: v for k, v in overrides.items() if k in train_config_fields}
    train_config_overrides["device"] = device

    model_config = TransformerConfig(**model_config_overrides)
    train_config_overrides.setdefault("model_name", "tuner")
    train_config_overrides.setdefault("model_version", "v0")
    train_config_overrides.setdefault("warmup_iters", 100)
    train_config = TrainConfig(**train_config_overrides)  # type: ignore

    print(f"model_config={model_config}")
    print(f"train_config={train_config}")
    model = create_random_model(
        model_config, action_vocab_size=vocab_size, num_players=num_players, seed=42, device=device
    )

    model, trainer = train_model(model, dataset_paths, train_config, device=device, n_max_context=n_max_context)
    loss_dict = trainer.estimate_loss()
    loss_dict = {k: float(v) for k, v in loss_dict.items()}

    elapsed = time.time() - t0
    print(
        f"## train_loss: {loss_dict['train']:.4f}, val_loss: {loss_dict['val']:.4f}, Time taken: {elapsed}s, val_policy_loss: {loss_dict['val_policy_loss']:.4f}, val_value_loss: {loss_dict['val_value_loss']:.4f}, overrides={overrides}"
    )
    return loss_dict, elapsed, model


class Tuner:
    """Class to automate the choice of model hyperparameters to reduce loss."""

    def __init__(
        self,
        fixed_params: dict[str, Any],
        initial_params: dict[str, Any],
        tune_options: dict[str, list[Any]],
        computed_tune_options: dict[str, Callable[[dict[str, Any]], list[Any]]],
        cache_version: str,
        target_improvement_per_minute: float = 0.0,
        initialize_from_best_model: bool = True,
        save_trained_models: bool = True,
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
            raise ValueError(
                f"Duplicate keys found in tune_keys and comuted_tune_keys -> {tune_keys & computed_tune_keys}"
            )
        if fixed_keys & all_tune_keys:
            raise ValueError(f"Can't tune fixed keys -> {fixed_keys & all_tune_keys}")
        if fixed_keys & initial_params_keys:
            raise ValueError(f"Duplicate fixed and initial keys -> {fixed_keys & initial_params_keys}")
        if all_tune_keys != initial_params_keys:
            raise ValueError(
                f"all_tune_keys != initial_params_keys. Added: {initial_params_keys - all_tune_keys}. Removed: {all_tune_keys - initial_params_keys}"
            )

        self.all_tune_keys = all_tune_keys
        self.target_improvement_per_minute = target_improvement_per_minute
        self.target_improvement_per_second = target_improvement_per_minute / 60
        self.save_trained_models = save_trained_models

        # Load cache file.
        self.model_cache_root = f"models/cache/{cache_version}"
        os.makedirs(self.model_cache_root, exist_ok=True)
        self.cache_path = f"result_cache-v{cache_version}.json"
        self.result_cache = json.load(open(self.cache_path)) if os.path.exists(self.cache_path) else {}

        self.best_hparams_path = f"best_hparams-v{cache_version}.json"
        self.best_model_trajectory = []

        if initialize_from_best_model and self.result_cache:
            best_cache_entry = min(
                self.result_cache.items(),
                key=lambda kv: kv[1]["val"] + kv[1]["elapsed"] * self.target_improvement_per_second,
            )
            best_cache_params = dict(ast.literal_eval(best_cache_entry[0]))
            initial_params = best_cache_params
            new_initial_params_keys = initial_params.keys()
            if initial_params_keys != new_initial_params_keys:
                raise ValueError(
                    f"initial_params_keys != new_initial_params_keys after loadding best params. Added: {new_initial_params_keys - initial_params_keys}. Removed: {initial_params_keys - new_initial_params_keys}"
                )

        self.current_params = initial_params.copy()
        self.current_params.update(self.fixed_params)

        self.best_params = None
        self.best_loss = None
        self.generation = 0

    def _save_model(self, param_key_hash, model):
        path = f"{self.model_cache_root}/{param_key_hash}.pt"
        if self.save_trained_models:
            torch.save(model, path)

    def _load_model(self, param_key_hash):
        path = f"{self.model_cache_root}/{param_key_hash}.pt"
        if os.path.exists(path):
            # TODO: weights_only=False has security issues ... we trust the source so fine for now.
            return torch.load(path, weights_only=False)
        return None

    def load_best_model(self):
        param_hash = self.best_loss_dict["param_hash"]
        model = self._load_model(param_hash)
        return model

    def _save_result_cache(self):
        json.dump(self.result_cache, open(self.cache_path, "w"))

    def train_and_compute_loss(self, params, reload_model=False, name=None) -> tuple[float, float, dict, Any]:
        """Look up loss in cache, or train model to compute it and save to cache."""
        param_key = str(sorted((k, v) for (k, v) in params.items() if k in self.all_tune_keys))
        param_key_hash = hashlib.sha256(param_key.encode("utf-8")).hexdigest()
        if param_key in self.result_cache:
            loss_dict = self.result_cache[param_key]
            param_key_hash = loss_dict["param_hash"]
            model = self._load_model(param_key_hash) if reload_model else None
        else:
            try:
                print(f"Training {name or str(params)}")
                loss_dict, elapsed, model = train_with(**params)  # type: ignore
                loss_dict["elapsed"] = elapsed
                loss_dict["param_hash"] = param_key_hash
            except Exception as e:
                import traceback

                print(f"Error training with params {params}: error='{e}' traceback='{traceback.format_exc()}'")
                loss_dict = {"val": float("inf"), "elapsed": float("inf")}
                model = None
                # TODO: Do we want to rethrow here? Maybe make 'strict' optional?
                raise e

            self.result_cache[param_key] = loss_dict
            self._save_result_cache()
            if model:
                self._save_model(param_key_hash, model)
        return loss_dict["val"], loss_dict["elapsed"], loss_dict, model

    def calc_score(self, loss, elapsed):
        return loss + elapsed * self.target_improvement_per_second

    def maybe_update_best_param(self, loss, elapsed, params, loss_dict) -> bool:
        """If model is improved, add it to result_cache['best_model_trajectory']."""
        if self.best_loss is None:
            self.best_loss = loss
            self.best_loss_elapsed = elapsed
            self.best_params = params.copy()
            self.best_loss_dict = loss_dict.copy()
            self.best_model_trajectory.append(
                {
                    "change": [],
                    "loss": loss,
                    "loss_delta": 0,
                    "elapsed": elapsed,
                    "args": params.copy(),
                    "loss_dict": loss_dict.copy(),
                }
            )
            return True

        # lower loss is better, lower elapsed is better, so lower score is better.
        best_loss_score = self.calc_score(self.best_loss, self.best_loss_elapsed)
        current_loss_score = self.calc_score(loss, elapsed)
        if current_loss_score >= best_loss_score:
            return False

        prev_best_loss = self.best_loss
        prev_best_loss_elapsed = self.best_loss_elapsed
        prev_best_params = self.best_params

        self.best_loss = loss
        self.best_loss_elapsed = elapsed
        self.best_params = params.copy()
        self.best_loss_dict = loss_dict.copy()

        all_keys = sorted(set(self.best_params.keys()) | set(prev_best_params.keys()))
        changed_params = [
            {"k": k, "old": prev_best_params.get(k), "new": self.best_params.get(k)}
            for k in all_keys
            if prev_best_params.get(k) != self.best_params.get(k)
        ]

        self.best_model_trajectory.append(
            {
                "change": changed_params,
                "loss": loss,
                "loss_delta": prev_best_loss - loss,
                "elapsed": elapsed,
                "elapsed_delta": prev_best_loss_elapsed - elapsed,
                "args": params.copy(),
                "loss_dict": loss_dict.copy(),
            }
        )
        self._save_result_cache()
        return True

    def _recalculate_tunable_params(self, params) -> dict[str, Any]:
        params = params.copy()
        for k, fn in self.computed_tune_options.items():
            current_val = params[k]
            possible_vals = fn(params)
            # self.tune_options[k] = possible_vals
            # TODO: Find the closest option?
            if current_val in possible_vals:
                continue
            elif len(possible_vals) == 1:
                params[k] = possible_vals[0]
            elif possible_vals == sorted(possible_vals):
                # Chose a value one below where the target value would be inserted.
                for i in range(len(possible_vals)):
                    if possible_vals[i] > current_val:
                        break
                params[k] = possible_vals[max(0, i - 1)]
            else:
                raise ValueError(f"Current value {current_val} for {k} not in unsorted possible values {possible_vals}")

        return params

    def select_candidate_params(self) -> list[tuple[str, dict[str, Any]]]:
        """Create a list of potential hyperparameter configs to improve model"""
        score_stats_raw = defaultdict(list)
        for key, val in self.result_cache.items():
            parsed_key = ast.literal_eval(key)
            score = self.calc_score(val["val"], val["elapsed"])
            score_stats_raw["ALL"].append(score)
            for param_key, param_val in parsed_key:
                score_stats_raw[(param_key, param_val)].append(score)

        score_stats = {k: sum(v) / len(v) for k, v in score_stats_raw.items()}

        def calc_expected_score(params):
            default_score = score_stats["ALL"]
            return sum(
                score_stats.get((param_key, param_val), default_score) for param_key, param_val in params.items()
            ) / len(params)

        def is_allowed_change(candidate_values, best_value, candidate_value) -> bool:
            """Return truen if candidate_value is next to best_value in tunable_values"""
            pos1 = candidate_values.index(best_value)
            pos2 = candidate_values.index(candidate_value)
            return abs(pos1 - pos2) == 1

        # Include computed-tuned-options in search.
        candidate_list = []
        computed_tune_options = {k: fn(self.best_params) for k, fn in self.computed_tune_options.items()}
        all_tune_options = self.tune_options.copy()
        all_tune_options.update(computed_tune_options)

        for param_name, candidate_values in all_tune_options.items():
            for candidate_value in candidate_values:
                if not is_allowed_change(candidate_values, self.best_params[param_name], candidate_value):
                    continue
                candidate_params = self.best_params.copy()
                candidate_params[param_name] = candidate_value
                candidate_params = self._recalculate_tunable_params(candidate_params)
                expected_score = calc_expected_score(candidate_params)
                name = f"{param_name}: {self.best_params[param_name]} -> {candidate_value}"
                candidate_list.append((expected_score, name, candidate_params))

        sorted_candidates = sorted(candidate_list)
        result = [(name, params) for (score, name, params) in sorted_candidates]
        return result

    def autotune_smart(self, max_generations=20):
        """Intelligently choose which hyperparameters to tune next."""
        if self.best_loss is None:
            print("Using initial model as baseline.")
            recalculated_params = self._recalculate_tunable_params(self.current_params)
            loss, elapsed, loss_dict, model = self.train_and_compute_loss(recalculated_params, name="initial")
            self.maybe_update_best_param(loss, elapsed, recalculated_params, loss_dict)
            print(
                f"## Initial Model, loss={self.best_loss} elapsed={self.best_loss_elapsed}s, val_policy={self.best_loss_dict.get('val_policy', -1):.4f}, val_value={self.best_loss_dict.get('val_value', -1):.4f}"
            )

        generation = 0
        improved = False
        while generation < max_generations:
            candidate_params_list = self.select_candidate_params()
            print(
                f"## Searching generation {generation} with {len(candidate_params_list)} candidates, including {[k for (k, v) in candidate_params_list[:5]]}"
            )
            if not self._find_improvement(candidate_params_list):
                break
            generation += 1
            improved = True

        return improved, self.best_loss, self.best_loss_elapsed, self.best_params

    def _find_improvement(self, candidate_params_list):
        for name, params in candidate_params_list:
            loss, elapsed, loss_dict, model = self.train_and_compute_loss(params, name=name)
            is_improved = self.maybe_update_best_param(loss, elapsed, params, loss_dict)
            print(f"## improved: {is_improved}, loss={loss:.4f} elapsed={elapsed:.2f}s, mutation {name}")
            if is_improved:
                return True
        return False

    # TODO: Deprecated. delete.
    def autotune(self, num_generations=10) -> tuple[bool, float, float, dict[str, Any]]:
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
            print("Training initial model as baseline.")
            recalculated_params = self._recalculate_tunable_params(self.current_params)
            loss, elapsed, loss_dict, model = self.train_and_compute_loss(recalculated_params)
            self.maybe_update_best_param(loss, elapsed, recalculated_params, loss_dict)
            print(f"## Initial Model, loss={self.best_loss} elapsed={self.best_loss_elapsed}s")

        new_best_model_found = False
        for generation in range(1, num_generations + 1):
            self.generation = generation
            new_best_model_found_this_generation = False
            # TODO: Be smarter about chosing key order?
            #       Maybe order keys by how often they outperform the the current parameter choice given the score function?
            for param_name in self.all_tune_keys:
                prev_best_loss = self.best_loss
                prev_best_loss_elapsed = self.best_loss_elapsed
                print(f"## Tuning generation {self.generation}: {param_name}")
                is_improved = self.tune_hyperparameter(param_name)
                if is_improved:
                    print(
                        f"## Tuning generation {self.generation}: {param_name} improved, val={self.best_params[param_name]}, best={self.best_loss}, delta={prev_best_loss - self.best_loss} elapsed={self.best_loss_elapsed}s delta={prev_best_loss_elapsed - self.best_loss_elapsed}s"
                    )
                new_best_model_found_this_generation |= is_improved
            if not new_best_model_found_this_generation:
                print(f"No updates for generation {generation}, stopping")
                return new_best_model_found, self.best_loss, self.best_loss_elapsed, self.best_params
            new_best_model_found = True

        return new_best_model_found, self.best_loss, self.best_loss_elapsed, self.best_params

    def _tune_hyperparameter_range(self, params, param_name, val_list) -> bool:
        """Try each idx in idx_list for param. Return as soon as one fails to update best_params. Return True if any updates succeed."""
        best_updated = False

        for val in val_list:
            params[param_name] = val
            params = self._recalculate_tunable_params(params)
            loss, elapsed, loss_dict, model = self.train_and_compute_loss(params)
            if self.maybe_update_best_param(loss, elapsed, params, loss_dict):
                best_updated = True
            else:
                return best_updated
        return best_updated

    def tune_hyperparameter(self, param_name) -> bool:
        current_params = self._recalculate_tunable_params(self.best_params)
        current_idx = self.tune_options[param_name].index(current_params[param_name])

        value_current = self.tune_options[param_name][current_idx]
        values_up = self.tune_options[param_name][current_idx + 1 :]
        values_down = self.tune_options[param_name][:current_idx][::-1]
        print(f"Tuning {param_name}. Current->{value_current} Up->{values_up} Down->{values_down}")

        # Try tuning from current_idx updwrds.
        if self._tune_hyperparameter_range(current_params, param_name, values_up):
            return True
        # Try tuning from current_idx downwards.
        if self._tune_hyperparameter_range(current_params, param_name, values_down):
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
            if str_key == "best_model_trajectory":
                continue
            list_key = ast.literal_eval(str_key)
            for param_idx, (param_name, param_val) in enumerate(list_key):
                remaining_key = str(list_key[:param_idx] + list_key[param_idx + 1 :])
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
                    # print(f"## {param_name} {val1} {val2} {loss_dicts}")
                    stats_dict[(param_name, val1, val2)] = self._compute_stats(loss_dicts)
        return stats_dict

    def _compute_stats(self, loss_dicts):
        stats_dict = {
            "mean_val_1": np.mean([loss_dict[0]["val"] for loss_dict in loss_dicts]),
            "mean_val_2": np.mean([loss_dict[1]["val"] for loss_dict in loss_dicts]),
            "mean_val_delta": np.mean([loss_dict[0]["val"] - loss_dict[1]["val"] for loss_dict in loss_dicts]),
            "mean_elapsed_1": np.mean([loss_dict[0]["elapsed"] for loss_dict in loss_dicts]),
            "mean_elapsed_2": np.mean([loss_dict[1]["elapsed"] for loss_dict in loss_dicts]),
            "mean_elapsed_delta": np.mean(
                [loss_dict[0]["elapsed"] - loss_dict[1]["elapsed"] for loss_dict in loss_dicts]
            ),
            "std_val_1": np.std([loss_dict[0]["val"] for loss_dict in loss_dicts]),
            "std_val_2": np.std([loss_dict[1]["val"] for loss_dict in loss_dicts]),
            "std_elapsed_1": np.std([loss_dict[0]["elapsed"] for loss_dict in loss_dicts]),
            "std_elapsed_2": np.std([loss_dict[1]["elapsed"] for loss_dict in loss_dicts]),
        }
        # Assume normal distribution, Calculate how many standard deviations away from the mean the delta is.
        stats_dict["std_val_delta"] = stats_dict["mean_val_delta"] / stats_dict["mean_val_1"]
        stats_dict["std_elapsed_delta"] = stats_dict["mean_elapsed_delta"] / stats_dict["mean_elapsed_1"]

        stats_dict = {k: float(v) for k, v in stats_dict.items()}
        return stats_dict


def retune_model(
    initial_params,
    tune_options,
    param_overrides,
    tune_option_overrides,
    cache_version,
    num_generations=10,
    out_label=None,
):
    # Tune single epoch model.
    initial_params = initial_params.copy()
    tune_options = tune_options.copy()

    initial_params.update(param_overrides)
    tune_options.update(tune_option_overrides)

    tuner = Tuner(
        fixed_params={},
        initial_params=initial_params,
        tune_options=tune_options,
        computed_tune_options={},
        cache_version=cache_version,
    )
    tuner.autotune(num_generations=num_generations)

    best_model_result = tuner.result_cache["best_model_trajectory"][-1]
    best_model_params = best_model_result["args"]

    if out_label:
        with open(f"out_{out_label}.json", "w") as f:
            json.dump(tuner.result_cache["best_model_trajectory"], f)

    print("\n## Best model:")
    pprint(best_model_result)
    return tuner, best_model_params

import importlib
import sys
import re
import inspect

import torch


def reload_local_modules(name_regex="rgi.*", reload_globals=True, verbose=True):
    """Reload all modules matching the name regex."""
    reloaded_modules = {}

    caller_frame = inspect.currentframe().f_back  # type: ignore
    caller_globals = caller_frame.f_globals  # type: ignore

    for name, module in list(sys.modules.items()):
        if re.match(name_regex, name):
            importlib.reload(module)
            reloaded_modules[name] = module
            if verbose:
                print(f"reloaded {name}")

    # Find names in globals() that point to objects from reloaded modules
    if reload_globals:
        for global_name, global_obj in list(caller_globals.items()):
            if inspect.isclass(global_obj) or inspect.isfunction(global_obj):
                module_name = getattr(global_obj, "__module__", None)

                if module_name in reloaded_modules:
                    module = reloaded_modules[module_name]
                    try:
                        caller_globals[global_name] = getattr(module, global_name)
                        if verbose:
                            print(f"  -> Updated '{global_name}' in globals() from '{module_name}'")
                    except AttributeError:
                        print(f"  -> WARNING: Could not find '{global_name}' in reloaded '{module_name}'")


def detect_device(require_accelerator=True):
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Detected device: {device}")
    if require_accelerator and device == 'cpu':
        raise ValueError("No accelerator found")
    return device
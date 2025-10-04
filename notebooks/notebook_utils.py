import importlib
import sys
import re


def reload_local_modules(name_regex="rgi.*", verbose=True):
    """Reload all modules matching the name regex."""
    for name, module in list(sys.modules.items()):
        if re.match(name_regex, name):
            importlib.reload(module)
            if verbose:
                print(f"reloaded {name}")

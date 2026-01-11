"""
Numba compatibility layer.

Provides a jit decorator that uses numba when available,
otherwise falls back to a no-op decorator.
"""

try:
    from numba import jit
except ImportError:
    # Fallback: no-op decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        # Handle both @jit and @jit() syntax
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

__all__ = ['jit']

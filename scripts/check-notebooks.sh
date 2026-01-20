#!/bin/bash
# Check notebooks for type errors using ty
#
# Suppressed rules (notebook-specific issues that can't be fixed):
# - unresolved-reference: Variables defined in earlier notebook cells appear undefined
# - invalid-argument-type: Type inference issues with lambdas and Dataset types
# - call-non-callable: Optional type narrowing issues
# - missing-argument: kwargs unpacking not understood by type checker
# - possibly-unbound-attribute: Optional type narrowing issues
# - invalid-await: Type checker doesn't understand async notebook context

uv run ty check --extra-search-path notebooks \
    --ignore unresolved-reference \
    --ignore invalid-argument-type \
    --ignore call-non-callable \
    --ignore missing-argument \
    --ignore possibly-unbound-attribute \
    --ignore invalid-await \
    notebooks/*.ipynb

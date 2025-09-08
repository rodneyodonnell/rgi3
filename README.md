# RGI project repo

This is an experimental repository for building RL & LLM tools.


## Project Setup:
```
# Sync dependencies.
uv sync


# launch jupyter notebook.
uv run jupyter lab

# Reformat and run 'precommit' checks.
./scripts/format-and-run-checks.sh
```

## Subprojects:

### nanoGPT.fork
- This is a fork & wrapper for nanoGPT to make it easier to call from other tools.
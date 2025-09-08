echo "Running nbstripout"
uv run nbstripout $(find -iname *.ipynb)

echo "Running ruff-format"
uv run ruff format .

echo "Running ruff check"
uv run ruff check .

echo "Running type checker"
uv run ty check .

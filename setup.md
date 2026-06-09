# Poetry Usage

## Setup & Install

```powershell
# Install all dependencies (creates venv automatically)
poetry install

# Activate the virtual environment
poetry shell
```

## Running the Project

```powershell
# Run main.py inside the poetry env (no need to activate shell first)
poetry run python main.py --help

# Build a knowledge base
poetry run python -m knowledge.builder --name bluetooth --docs ./docs/

# Run analysis
poetry run python main.py -f reqs.txt --kb bluetooth
```

## Managing Dependencies

```powershell
# Add a new package (updates pyproject.toml + poetry.lock)
poetry add requests

# Remove a package
poetry remove requests

# Update all packages to latest allowed versions
poetry update

# Show installed packages
poetry show
```

## Sharing / Reproducing the Environment

```powershell
# Someone cloning the repo runs this to get the exact same versions
poetry install

# Export to requirements.txt (for environments that don't use Poetry)
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

## Key Concept

- **`pyproject.toml`** — version ranges you maintain (e.g., `openai = ">=1.0.0"`)
- **`poetry.lock`** — exact pinned versions Poetry resolved — commit this to git so all team members and CI get identical installs

The existing `venv/` folder can be deleted once you switch to Poetry — Poetry manages its own virtual environment.

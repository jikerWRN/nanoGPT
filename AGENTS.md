# Repository Guidelines

## Project Structure & Module Organization

The repository is intentionally small. `model.py` defines the GPT architecture, while `train.py`, `sample.py`, and `bench.py` provide training, inference, and performance entry points. Experiment settings live in `config/` as executable Python configuration files. Dataset download and preprocessing scripts are under `data/<dataset>/`; generated `.bin` and `.pkl` files are ignored. Images used by `README.md` belong in `assets/`. Root notebooks contain scaling and model-sizing experiments.

## Setup, Training, and Development Commands

Install dependencies in a virtual environment:

```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

Run the smallest end-to-end example with:

```bash
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py
python sample.py --out_dir=out-shakespeare-char
```

On CPU or Windows, add `--device=cpu --compile=False`; reduce `--max_iters` for a quick smoke test. Full GPT-2 training uses `torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py`. Run `python bench.py` when evaluating performance changes.

## Coding Style & Naming Conventions

Use four-space indentation and follow the existing compact, PEP 8-oriented Python style. Use `snake_case` for functions, variables, CLI/config keys, and filenames; use `PascalCase` for classes such as `GPTConfig` and `CausalSelfAttention`. Keep configuration files declarative and name them by action and target, for example `train_gpt2.py` or `eval_gpt2_large.py`. No formatter or linter is configured, so match surrounding code.

## Testing Guidelines

There is no automated test directory, test framework, or coverage threshold. Validate changes with a short Shakespeare training run and sampling pass. For model changes, check tensor shapes, loss evaluation, checkpoint loading, and both compiled and `--compile=False` paths where available. Do not commit generated checkpoints or datasets.

## Commit & Pull Request Guidelines

Recent history uses short imperative subjects, sometimes with Conventional Commit prefixes such as `fix:`. Keep each commit scoped and describe the observable change. Pull requests should explain motivation, implementation, and validation commands; link related issues and include benchmark numbers for performance work. Add sample output or loss comparisons when training behavior changes, and screenshots only for documentation or notebook visuals.

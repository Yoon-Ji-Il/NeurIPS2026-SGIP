# Source release notes

This tree was prepared separately from the original research workspace.
The original files, runs, and checkpoints were left in place.

## Included

- Python and shell source from the repository root, `config`, `dataset`,
  `model`, `object_selector`, `object_selector_new`, `scripts`, `utils`, and `tests`.
- Existing Markdown experiment notes, plus setup and data documentation.
- The SGIP overview figure in `docs/assets/`, supplied for the README.
- Dependency inventories, ignore rules, and a staged-file release check.

## Excluded

- All `.pth`, `.pt`, `.ckpt`, and other model/data binaries.
- The entire external VPoser snapshot, including its weights and source.
- Dataset payloads, generated scene graphs, teacher targets, selector CSVs,
  training/evaluation outputs, plots, logs, and Python caches.
- The nested `object_selector_new/mnt/user-data/outputs` duplicate.
- Stale root READMEs that claimed checkpoints were bundled, and a generated PDF.

## Portability changes

Machine-specific dataset/model defaults now use repository-relative locations
documented in `data.md`. Shell launchers resolve the checkout location and use
the active Python environment. Embedded private Ollama addresses now default
to localhost. The main GIMO paths support environment variables and CLI
overrides. No architecture, loss formula, or training hyperparameter was
intentionally changed by this packaging work.

The inherited KD configuration registered three teacher arguments twice,
which raised an `argparse` conflict during construction. It now overrides
their defaults without registering duplicate options. Teacher train/eval
launchers accept VPoser paths and forwarded CLI options, and their default
run directories agree.

## Limits

The original environment lock, paper title/authors/citation, and project-wide
license were not supplied. They are not invented here. Existing source notices
are retained. Dependencies are an inventory, not a tested full environment lock.

Validation of packaging and the existing lightweight tests does not establish
end-to-end trainability or reproduce paper metrics. Full training requires
external datasets, body-model assets, the compatible VPoser implementation,
and a working PointNet2 build.

## Validation performed

- Existing unit tests: 9 passed (PyTorch 2.2.0 CPU execution; no model training).
- Python syntax and shell syntax checks passed for all released scripts.
- All five configuration classes constructed and parsed their defaults.
- GIMO environment defaults and explicit CLI overrides were checked.
- Teacher train/eval launchers were exercised with a mocked Python executable,
  including paths containing spaces, forwarded arguments, and a missing run.
- Representative checkpoint, data, and credential paths were confirmed ignored.
- Staged-file checks and Git whitespace checks passed.

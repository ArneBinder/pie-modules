# pie-modules

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://github.com/ChristophAlt/pytorch-ie"><img alt="PyTorch-IE" src="https://img.shields.io/badge/-PyTorch--IE-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

[![PyPI](https://img.shields.io/pypi/v/pie-modules.svg)][pypi status]
[![Tests](https://github.com/arnebinder/pie-modules/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/arnebinder/pie-modules/branch/main/graph/badge.svg)][codecov]
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

Model-, taskmodule-, and metric-implementations for [PyTorch-IE](https://github.com/ChristophAlt/pytorch-ie).

Available models:

- [SimpleSequenceClassificationModel](src/pie_modules/models/simple_sequence_classification.py)
- [SequenceClassificationModel](src/pie_modules/models/sequence_classification.py)
- [TokenClassificationModelWithSeq2SeqEncoderAndCrf](src/pie_modules/models/token_classification_with_seq2seq_encoder_and_crf.py)
- [SimpleExtractiveQuestionAnsweringModel](src/pie_modules/models/simple_extractive_question_answering.py)

Available taskmodules:

- [RETextClassificationWithIndicesTaskModule](src/pie_modules/taskmodules/re_text_classification_with_indices.py)
- [TokenClassificationTaskModule](src/pie_modules/taskmodules/token_classification.py)
- [ExtractiveQuestionAnsweringTaskModule](src/pie_modules/taskmodules/extractive_question_answering.py)

Available metrics:

- [SpanLengthCollector](src/pie_modules/metrics/span_length_collector.py)
- [SQuADF1](src/pie_modules/metrics/squad_f1.py)

## Setup

```bash
pip install pie-modules
```

To install the latest version from GitHub:

```bash
pip install git+https://git@github.com/ArneBinder/pie-modules.git
```

## Development

### Setup

1. This project is build with [Poetry](https://python-poetry.org/). See here for [installation instructions](https://python-poetry.org/docs/#installation).
2. Get the code and switch into the project directory:
   ```bash
   git clone https://github.com/ArneBinder/pie-modules
   cd pie-modules
   ```
3. Create a virtual environment and install the dependencies:
   ```bash
   poetry install
   ```

Finally, to run any of the below commands, you need to activate the virtual environment:

```bash
poetry shell
```

Note: You can also run commands in the virtual environment without activating it first: `poetry run <command>`.

### Code Formatting, Linting and Static Type Checking

```bash
pre-commit run -a
```

### Testing

run all tests with coverage:

```bash
pytest --cov --cov-report term-missing
```

### Releasing

1. Create the release branch:
   `git switch --create release main`
2. Increase the version:
   `poetry version <PATCH|MINOR|MAJOR>`,
   e.g. `poetry version patch` for a patch release. If the release contains new features, or breaking changes,
   bump the minor version (this project has no main release yet). If the release contains only bugfixes, bump
   the patch version. See [Semantic Versioning](https://semver.org/) for more information.
3. Commit the changes:
   `git commit --message="release <NEW VERSION>" pyproject.toml`,
   e.g. `git commit --message="release 0.13.0" pyproject.toml`
4. Push the changes to GitHub:
   `git push origin release`
5. Create a PR for that `release` branch on GitHub.
6. Wait until checks passed successfully.
7. Merge the PR into the main branch. This triggers the GitHub Action that creates all relevant release
   artefacts and also uploads them to PyPI.
8. Cleanup: Delete the `release` branch. This is important, because otherwise the next release will fail.

[black]: https://github.com/psf/black
[codecov]: https://app.codecov.io/gh/arnebinder/pie-modules
[pre-commit]: https://github.com/pre-commit/pre-commit
[pypi status]: https://pypi.org/project/pie-modules/
[tests]: https://github.com/arnebinder/pie-modules/actions?workflow=Tests

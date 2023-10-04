# pie-models

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://github.com/ChristophAlt/pytorch-ie"><img alt="PyTorch-IE" src="https://img.shields.io/badge/-PyTorch--IE-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

[![PyPI](https://img.shields.io/pypi/v/pie-models.svg)][pypi status]
[![Tests](https://github.com/arnebinder/pie-models/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/arnebinder/pie-models/branch/main/graph/badge.svg)][codecov]
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

Model and Taskmodule implementations for [PyTorch-IE](https://github.com/ChristophAlt/pytorch-ie).

## Setup

```bash
pip install git+https://github.com/ArneBinder/pie-models.git
```

or

```bash
pip install git+ssh://git@github.com/ArneBinder/pie-models.git
```

or add this to your `requirements.txt`:

```
git+https://github.com/ArneBinder/pie-models
```

Note: You can specify a specific version by appending a version tag `@v<version>` to the URL,
e.g. `git+https://github.com/ArneBinder/pie-models@v0.6.0`.

## Development

### Setup

```bash
git clone https://github.com/ArneBinder/pie-models
cd pie-models
pip install -e ".[dev]"
```

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

1. create a branch `release` from the `main` branch
2. bump the version in `setup.py`. If the release contains new features, or breaking changes, bump the minor version (this project has no main release yet). If the release contains only bugfixes, bump the patch version. See [Semantic Versioning](https://semver.org/) for more information.
3. commit and push the changes
4. create a pull request from `release` to `main`
5. wait for the CI to pass
6. merge the pull request and delete the `release` branch (this is important, because otherwise the next release will fail)
7. create a new release on GitHub via the "Releases" tab and click on "Draft a new release".
   1. Click on "Choose a tag" and create a new one which should be the same as the version in `setup.py`, but prefixed with `v`, e.g. `v0.6.1` for version `0.6.1`.
   2. You can choose an appropriate release title.
   3. Click on "Generate release notes" to generate the release notes from the pull request descriptions.
   4. When everything looks fine, click on "Publish release" to publish the release.

[black]: https://github.com/psf/black
[codecov]: https://app.codecov.io/gh/arnebinder/pie-models
[pre-commit]: https://github.com/pre-commit/pre-commit
[pypi status]: https://pypi.org/project/pie-models/
[tests]: https://github.com/arnebinder/pie-models/actions?workflow=Tests

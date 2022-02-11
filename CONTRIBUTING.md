# Contributing to HOLA

## Filing issues

When filing an issue, make sure to answer these five questions:

1. What version of Python are you using (`python --version`)?
2. What did you do?
3. What did you expect to see?
4. What did you see instead?

## Report a Bug

Open an issue. Please include descriptions of the following:

- Observations
- Expectations
- Steps to reproduce

## Contributing code

In general, this project follows Python project conventions. Please make sure you've linted, formatted, and run your
tests before submitting a patch. To set up your working environment run `make env`. To format your code run `make fmt`,
to lint it, run `make lint`, to test it, run `make test`.

## Contribute a Bug Fix

- Report the bug first
- Create a pull request for the fix

## Suggest a New Feature

- Create a new issue to start a discussion around new topic. Label the issue as `new-feature`

## Developer guidelines

### Environment

Run `make env` to build the dev/research HOLA environment. Example using `conda`:

```bash
conda create -n hola python=3.8 -y
conda activate hola
make env
```

### Platforms

#### Linux and WSL
On Linux and WSL everything should work out of the box.

#### Windows
On Windows, please use `Git Bash` and install `make` as a one off.

#### Mac
HOLA relies heavily on the GNU version of `find` and `xargs`. Mac, by default, uses BSD rather than GNU so please
run `brew install findutils` as a one-off to bring the GNU versions of these utils.

### Formatting
To format the code run `make fmt`.

### Linting
To lint the code run `make lint`.

### Testing
To test the code and see the coverage run `make test`.

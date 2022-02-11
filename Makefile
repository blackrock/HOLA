ONPY=hola/ tests/ benchmarks/ analysis/
ONNB=notebooks/
ONSH=*.sh
ONYML=.azuredevops *.yaml

BLACK_FLAGS=--config pyproject.toml
ISORT_FLAGS=--settings-path pyproject.toml
DOCFORMATTER_FLAGS=--wrap-summaries=120 --wrap-descriptions=120
AUTOFLAKE_FLAGS=--remove-all-unused-imports
MYPY_FLAGS=--config-file mypy.ini
BANDIT_FLAGS=--configfile pyproject.toml
YAMLLINT_FLAGS=--config-file yamllint.yaml
PYTEST_FLAGS=-c pyproject.toml
#FLAKE8_NB_FLAGS=--config .flake8_nb

fmt: fmt-py # fmt-nb # fmt-sh
fmt-py: black isort docformatter autoflake flynt
fmt-nb: nbstripout black-nb
#fmt-sh: shfmt

lint: lint-sh lint-yml lint-py # lint-nb
lint-py: docformatter-check isort-check black-check autoflake-check flake8 bandit mypy pylint
lint-nb: black-nb-check flake8-nb
lint-sh: shellcheck # shfmt-check
lint-yml: yamllint

ifneq ($(shell command -v gfind),)  # is MacOS (mostly)
gnu_find=gfind
else
gnu_find=find
endif

ifneq ($(shell command -v gxargs),)  # is MacOS (mostly)
gnu_xargs=gxargs
else
gnu_xargs=xargs
endif

# FORMAT ---------------------------------------------------------------------------------------------------------------
.PHONY: black
black:
	python -m black $(BLACK_FLAGS) $(ONPY)

.PHONY: isort
isort:
	python -m isort $(ISORT_FLAGS) $(ONPY)

.PHONY: docformatter
docformatter:
	python -m docformatter --in-place $(DOCFORMATTER_FLAGS) -r $(ONPY)

.PHONY: autoflake
autoflake:
	python -m autoflake --in-place $(AUTOFLAKE_FLAGS) -r $(ONPY)

.PHONY: flynt
flynt:
	python -m flynt $(ONPY)

.PHONY: nbstripout
nbstripout:
	$(gnu_find) $(ONNB) -type f -name "*.ipynb" -print | $(gnu_xargs) python -m nbstripout --strip-empty-cells

.PHONY: black-nb
black-nb:
	python -m black $(BLACK_FLAGS) $(ONNB)

.PHONY: shfmt
shfmt:
	shfmt -w $(SHFMT_FLAGS) $(ONSH)

# LINT -----------------------------------------------------------------------------------------------------------------
.PHONY: black-check
black-check:
	python -m black --check $(BLACK_FLAGS) $(ONPY)

.PHONY: docformatter-check
docformatter-check:
	python -m docformatter $(DOCFORMATTER_FLAGS) -r $(ONPY) && \
	python -m docformatter --check $(DOCFORMATTER_FLAGS) -r $(ONPY)

.PHONY: isort-check
isort-check:
	python -m isort --diff --color --check-only $(ISORT_FLAGS) $(ONPY)

.PHONY: autoflake-check
autoflake-check:
	python -m autoflake --in-place $(AUTOFLAKE_FLAGS) -r $(ONNB)

.PHONY: flake8
flake8:
	python -m flake8 $(FLAKE8_FLAGS) $(ONPY)

.PHONY: pylint
pylint:
	python -m pylint $(PYLINT_FLAGS) $(ONPY)

.PHONY: bandit
bandit:
	python -m bandit $(BANDIT_FLAGS) -r $(ONPY)

.PHONY: black-nb-check
black-nb-check:
	python -m black --check $(BLACK_FLAGS) $(ONNB)

.PHONY: flake8-nb
flake8-nb:
	$(gnu_find) $(ONNB) -type f -iname "*.ipynb" | $(gnu_xargs) python -m flake8_nb $(FLAKE8_NB_FLAGS)

.PHONY: shfmt-check
shfmt-check:
	shfmt -d $(SHFMT_FLAGS) $(ONSH)

.PHONY: shellcheck
shellcheck:
	$(gnu_find) $(ONSH) -type f -iname "*.sh" | $(gnu_xargs) shellcheck $(SHELLCHECK_FLAGS)

.PHONY: yamllint
yamllint:
	python -m yamllint $(YAMLLINT_FLAGS) $(ONYML)

# TYPE CHECK -----------------------------------------------------------------------------------------------------------
.PHONY: mypy
mypy:
	python -m mypy $(MYPY_FLAGS) $(ONPY)

# TEST -----------------------------------------------------------------------------------------------------------------
.PHONY: test
test:
	python -m pytest $(PYTEST_FLAGS) $(ONPY)


# CLEAN ----------------------------------------------------------------------------------------------------------------
.PHONY: clean-pyc
clean-pyc:
	$(gnu_find) . -name *.pyc | $(gnu_xargs) rm -f && $(gnu_find) . -name *.pyo | $(gnu_xargs) rm -f;

.PHONY: clean-build
clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info

# ENV  -----------------------------------------------------------------------------------------------------------------
.PHONY: env
env:
	python -m pip install --upgrade pip setuptools $(extra_pip_flags)
	python -m pip install --upgrade -r requirements.txt -r requirements-dev.txt -r requirements-mypy.txt -r requirements-research.txt $(extra_pip_flags)

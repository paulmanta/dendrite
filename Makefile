# -----------------------------------------------------------------------------
# Custom functions
# -----------------------------------------------------------------------------

find = $(wildcard $1/*$2) $(foreach d,$(wildcard $1/*),$(call find,$d,$2))
run-integration-test = (export PYTHONPATH=. && python3 -m tests.integration.$1.main)

# -----------------------------------------------------------------------------
# Directory names
# -----------------------------------------------------------------------------

PROJ_DIR             = app
DOCS_DIR             = docs
TEST_DIR             = tests
UNIT_TEST_DIR        = $(TEST_DIR)/unit
INTEGRATION_TEST_DIR = $(TEST_DIR)/integration

# -----------------------------------------------------------------------------
# Recursively build lists of all Python-specific files and directories.
# -----------------------------------------------------------------------------

PY_FILES     := $(call find,$(PROJ_DIR),.py) $(call find,$(TEST_DIR),.py)
PYC_FILES    := $(call find,.,.pyc)
PYCACHE_DIRS := $(call find,.,__pycache__)

# -----------------------------------------------------------------------------
# Build rules
# -----------------------------------------------------------------------------

.PHONY: clean
clean:
	@rm -f $(PYC_FILES)
	@rm -rf $(PYCACHE_DIRS) $(DOCS_DIR)/app/build $(DOCS_DIR)/app/source/api

.PHONY: unit
unit:
	@nosetests -w . $(UNIT_TEST_DIR)

.PHONY: test-iris
test-iris:
	@$(call run-integration-test,iris)

.PHONY: test-autovit
test-autovit:
	@$(call run-integration-test,autovit)

.PHONY: pep8
pep8:
	@pep8 $(PY_FILES)

.PHONY: doc
doc:
	@sphinx-apidoc -feMT -o $(DOCS_DIR)/app/source/api $(PROJ_DIR)
	@sphinx-build -b html $(DOCS_DIR)/app/source $(DOCS_DIR)/app/build

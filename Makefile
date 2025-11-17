
# general
mkfile_path := $(abspath $(firstword $(MAKEFILE_LIST)))
current_dir := $(notdir $(patsubst %/,%,$(dir $(mkfile_path))))
current_abs_path := $(subst Makefile,,$(mkfile_path))

# pipeline constants
# PROJECT_NAME
project_name := "2025-autumn-city-of-chicago-energy"
project_dir := "$(current_abs_path)"

# environment variables
include .env

# Set SSH agent configuration based on OS
ifeq ($(shell uname -s),Darwin)
    DOCKER_SSH_AUTH_SOCK := /run/host-services/ssh-auth.sock
else ifeq ($(shell uname -s),Linux)
    DOCKER_SSH_AUTH_SOCK := $(SSH_AUTH_SOCK)
else
    $(error Unsupported operating system. Please set DOCKER_SSH_AUTH_SOCK manually)
endif

export DOCKER_SSH_AUTH_SOCK

# Check required environment variables
ifeq ($(DATA_DIR),)
    $(error DATA_DIR must be set in .env file)
endif


# Build Docker image 
# Global mount for data directory
mount_data := -v $(DATA_DIR):/project/data

.PHONY: build-only run-interactive run-notebooks test-pipeline clean

# Build Docker image 
build-only: 
	docker compose build

run-interactive: build-only
	docker compose run -it --rm $(mount_data) $(project_name) /bin/bash

run-notebooks: build-only
	docker compose run --rm -p 8888:8888 -t $(mount_data) $(project_name) uv run jupyter lab --port=8888 --ip='*' --no-browser --allow-root --log-level=INFO

run-streamlit: build-only
	docker compose run --rm -p 8501:8501 -t $(mount_data) $(project_name) uv run streamlit run src/dashboard/Introduction.py --server.port=8501 --server.address=0.0.0.0

test-pipeline: build-only
	docker compose run --rm $(mount_data) $(project_name) uv run python src/utils/pipeline_example.py

clean:
	docker compose down --rmi all --volumes --remove-orphans
	docker image prune -f



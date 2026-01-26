test:
	@echo Running tests
	@pytest --cov=src tests/

lint:
	@echo Running code linters
	@echo Running ruff
	@ruff check src tests

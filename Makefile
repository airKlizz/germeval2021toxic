# Check that source code meets quality standards

quality:
	black --check --line-length 119 --target-version py37 .
	isort --check-only .
	flake8 --max-line-length 119

# Format source code automatically

style:
	black --line-length 119 --target-version py37 .
	isort .
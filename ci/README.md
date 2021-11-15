## For a future CI

The pre-commits are specified in the .pre-commit-config.yaml file. 

To install them:
```
pip install pre-commit
pre-commit install
```

To run them:
```
pre-commit run --all-files
```

Note: the unit tests (pytest) are included in the pre-commits

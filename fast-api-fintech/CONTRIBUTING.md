# Contributing to FastAPI Fintech Application

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/python-study.git`
3. Create a feature branch: `git checkout -b feature/amazing-feature`
4. Make your changes
5. Test your changes
6. Commit your changes: `git commit -m 'Add some amazing feature'`
7. Push to the branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment variables
cp .env.example .env

# Run the application
python -m app.main
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints for all functions
- Add docstrings to all public functions and classes
- Keep functions small and focused
- Write descriptive variable names

### Formatting

```bash
# Format code with black
black app/ tests/

# Sort imports with isort
isort app/ tests/

# Lint code with flake8
flake8 app/ tests/

# Type check with mypy
mypy app/
```

## Testing

- Write tests for all new features
- Maintain test coverage above 80%
- Use pytest for testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/unit/test_banking.py
```

## Commit Messages

Follow conventional commit format:

- `feat: Add new feature`
- `fix: Fix bug`
- `docs: Update documentation`
- `style: Format code`
- `refactor: Refactor code`
- `test: Add tests`
- `chore: Update dependencies`

## Pull Request Process

1. Update README.md with details of changes if applicable
2. Update documentation in the `docs/` directory
3. Add tests for new functionality
4. Ensure all tests pass
5. Update the CHANGELOG.md
6. Request review from maintainers

## Code Review

- Be respectful and constructive
- Address all review comments
- Keep discussions focused on code
- Be open to feedback

## Bug Reports

When reporting bugs, include:

- Description of the issue
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, etc.)
- Screenshots if applicable

## Feature Requests

When requesting features, include:

- Use case description
- Proposed solution
- Alternative solutions considered
- Additional context

## Questions?

Feel free to open an issue for any questions or concerns.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

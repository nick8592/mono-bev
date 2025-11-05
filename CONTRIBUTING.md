# Contributing to Mono-BEV

Thank you for your interest in contributing to the Monocular 2D-to-BEV Detection Pipeline! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, GPU, etc.)
- Relevant code snippets or error messages

### Suggesting Enhancements

Enhancement suggestions are welcome! Please open an issue with:

- A clear description of the enhancement
- Use cases and benefits
- Possible implementation approach
- Any potential drawbacks

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following the code style guidelines
3. **Add tests** if applicable
4. **Update documentation** if needed
5. **Commit your changes** with clear commit messages
6. **Push to your fork** and submit a pull request

## 📝 Code Style Guidelines

### Python Code Style

We follow [PEP 8](https://pep8.org/) with some modifications:

```python
# Good: Clear, documented, type-hinted
def process_detections(
    detections: List[Dict],
    threshold: float = 0.5
) -> List[Dict]:
    """
    Process detections above confidence threshold.
    
    Args:
        detections: List of detection dictionaries
        threshold: Minimum confidence score
        
    Returns:
        Filtered list of detections
    """
    return [d for d in detections if d['confidence'] >= threshold]
```

### Key Principles

1. **Type Hints**: Use type hints for function signatures
2. **Docstrings**: All public functions and classes must have docstrings
3. **Line Length**: Maximum 100 characters (not strict)
4. **Imports**: Group by standard library, third-party, local
5. **Naming**: 
   - `snake_case` for functions and variables
   - `PascalCase` for classes
   - `UPPERCASE` for constants

### Documentation

- Update docstrings when modifying functions
- Add comments for complex logic
- Update README.md if adding features
- Keep documentation in sync with code

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_pipeline.py

# Run with coverage
python -m pytest --cov=src tests/
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Use descriptive test function names
- Include both positive and negative test cases

Example:

```python
import pytest
from src.models.detector import YOLODetector

def test_detector_initialization():
    """Test YOLODetector can be initialized with valid config."""
    config = {'detector': {'model_name': 'yolo11n', 'device': 'cpu'}}
    detector = YOLODetector(config)
    assert detector is not None

def test_detector_invalid_image():
    """Test detector handles invalid input gracefully."""
    config = {'detector': {'model_name': 'yolo11n', 'device': 'cpu'}}
    detector = YOLODetector(config)
    
    with pytest.raises(ValueError):
        detector.detect(None)
```

## 🌿 Git Workflow

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test additions/modifications

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or modifying tests
- `chore`: Maintenance tasks

Examples:

```bash
feat(detector): add batch processing support

Add batch detection capability to YOLODetector class
to improve throughput for multi-image inference.

Closes #123
```

```bash
fix(visualizer): correct BEV coordinate transformation

Fix rotation matrix calculation in _get_box_corners
that was causing incorrect object orientations.
```

## 🏗️ Development Setup

### Environment Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/mono-bev.git
cd mono-bev

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks (optional)
pre-commit install
```

### Development Dependencies

Additional packages for development:

```bash
pip install pytest pytest-cov black flake8 mypy pre-commit
```

## 📋 Checklist for Pull Requests

Before submitting a PR, ensure:

- [ ] Code follows the style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages follow conventions
- [ ] No unnecessary files included
- [ ] Branch is up to date with main

## 🔍 Code Review Process

1. **Automated Checks**: GitHub Actions will run tests and linting
2. **Maintainer Review**: A maintainer will review your code
3. **Feedback**: Address any feedback or requested changes
4. **Approval**: Once approved, your PR will be merged
5. **Recognition**: You'll be added to contributors!

## 💡 Areas for Contribution

We especially welcome contributions in:

- **Performance Optimization**: Speed up inference or training
- **New Features**: Multi-camera support, tracking, etc.
- **Testing**: Increase test coverage
- **Documentation**: Improve clarity and examples
- **Bug Fixes**: Fix reported issues
- **Visualization**: Better plotting and analysis tools

## 🎓 Learning Resources

If you're new to the project:

1. Read the [Architecture Guide](docs/ARCHITECTURE.md)
2. Check the [API Reference](docs/API.md)
3. Review existing issues and PRs
4. Start with "good first issue" labels

## 📞 Getting Help

- **Issues**: Open a GitHub issue for bugs or questions
- **Discussions**: Use GitHub Discussions for general questions
- **Email**: Contact maintainers directly for sensitive matters

## 🌟 Recognition

Contributors will be:

- Listed in the project README
- Acknowledged in release notes
- Given appropriate credit in documentation

Thank you for contributing to Mono-BEV! 🚀

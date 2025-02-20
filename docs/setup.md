# Setup Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for version control)

## Installation Steps

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd merging_app
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration**
   Create a `.env` file in the project root:
   ```
   APP_ENV=development
   DASK_PARTITIONS=10
   MAX_FILE_SIZE_MB=500
   CORS_ORIGINS=*
   ```

5. **Run Tests**
   ```bash
   pytest tests/
   ```

6. **Start Application**
   ```bash
   streamlit run app.py
   ```

## Development Setup

1. **Install Development Dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Setup Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

3. **Generate Documentation**
   ```bash
   cd docs
   make html
   ```

## Troubleshooting

If you encounter any issues:

1. Verify Python version:
   ```bash
   python --version
   ```

2. Check virtual environment activation:
   ```bash
   which python  # On Windows use: where python
   ```

3. Verify package installation:
   ```bash
   pip list
   ```

## Security Notes

- Keep your virtual environment active while working
- Never commit sensitive data or credentials
- Regularly update dependencies
- Follow security best practices

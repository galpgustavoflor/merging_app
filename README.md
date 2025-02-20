# Dataset Comparison and Validation Tool

A powerful Streamlit-based application for comparing, validating, and analyzing datasets with an intuitive user interface.

## Features

- File upload support for CSV and Excel files
- Interactive dataset mapping and key matching
- Customizable validation rules
- Data quality assessment and reporting
- Real-time visualization of comparison results
- Support for large datasets using Dask
- JSON-based configuration for mapping and validation rules

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd merging_app
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
streamlit run app.py --server.maxUploadSize=500
```

2. Follow the step-by-step process in the UI:
   - Upload source and target files
   - Define mapping rules
   - Configure validation rules
   - Execute matching and validation
   - Review results and download reports

## Configuration

### Mapping Rules

Create a JSON file with mapping configuration:
```json
{
    "key_source": ["id"],
    "key_target": ["reference_id"],
    "mappings": {
        "amount": {
            "destinations": ["value"],
            "function": "Direct Match",
            "transformation": null
        }
    }
}
```

### Validation Rules

Create a JSON file with validation rules:
```json
{
    "amount": {
        "validate_nulls": true,
        "validate_range": true,
        "min_value": 0,
        "max_value": 1000000
    }
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

# Merging App

This project is a Streamlit application for data merging.

## Prerequisites

- Docker

## Build the Docker Image

To build the Docker image, run the following command:

```sh
docker build -t merging_app .
```

## Run the Docker Container

To run the Docker container, use the following command:

```sh
docker run -p 8501:8501 merging_app
```

This will start the Streamlit application, and it will be accessible at `http://localhost:8501`.

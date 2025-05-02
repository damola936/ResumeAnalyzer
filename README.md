# Resume Analyzer

## Overview
Resume Analyzer is a Streamlit-based application designed to analyze resumes. It leverages a Conda environment for dependency management and is containerized using Docker for easy deployment.

## Features
- Analyze resumes using a user-friendly web interface.
- Built with Python and Streamlit.
- Fully containerized with Docker for consistent deployment.

## Project Structure
```
ResumeAnalyzer/
├── Dockerfile          # Docker configuration for containerizing the app
├── environment.yml     # Conda environment configuration
├── LICENSE             # License for the project
├── README.md           # Project documentation
├── requirements.txt    # Python dependencies (if applicable)
├── app/                # Application source code
│   └── app.py          # Main Streamlit application
├── docker/             # Additional Docker-related files (if any)
```

## Getting Started

### Prerequisites
- Docker installed on your system.
- (Optional) Conda installed for local development.

### Build and Run with Docker
1. Build the Docker image:
   ```bash
   docker build -t resume-analyzer .
   ```
2. Run the Docker container:
   ```bash
   docker run -p 8501:8501 resume-analyzer
   ```
3. Open your browser and navigate to `http://localhost:8501` to access the application.

### Local Development
1. Create the Conda environment:
   ```bash
   conda env create -f environment.yml
   ```
2. Activate the environment:
   ```bash
   conda activate resume-analyzer
   ```
3. Run the application:
   ```bash
   streamlit run app/app.py
   ```

## License
This project is licensed under the terms of the LICENSE file.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## Contact
For questions or feedback, please contact the project maintainer.
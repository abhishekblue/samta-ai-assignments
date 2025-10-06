# AI-ML Assignments

This repository contains solutions and implementations for various AI and Machine Learning assignments. The projects cover different aspects of AI/ML, including Natural Language Processing (NLP) and Retrieval-Augmented Generation (RAG).

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Assignments](#assignments)
  - [Task 1: Document Processing and RAG](#task-1-document-processing-and-rag)
  - [Task 2: Document Processing and RAG with Text and Word File](#task-2-add-description-for-task-2-if-available)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

## Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/AI-ML_assignments_samta.git
    cd AI-ML_assignments_samta
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure environment variables:**
    Create a `.env` file in the root directory and add necessary API keys or configurations. For example:
    ```
    GOOGLE_API_KEY=your_gemini_api_key_here
    ```

## Project Structure

The repository is organized as follows:

```
.
├── .env                      # Environment variables
├── README.md                 # Project overview and instructions
├── requirements.txt          # Python dependencies
├── task1_main.py             # Main script for Task 1
├── task2_main.py             # Main script for Task 2
├── data/                     # Directory for input data files
│   ├── extracted_text.docx   # Example DOCX document
│   └── sample.pdf            # Example PDF document
└── src/                      # Source code directory
    ├── __init__.py           # Python package initializer
    ├── config.py             # Configuration settings
    ├── document_processor.py # Handles document parsing and processing
    ├── rag_engine.py         # Implements the RAG logic
    └── vector_store.py       # Manages vector database operations
```
## Branches

- `main` - Latest stable version
- `feature/error-handling` - Enhanced error handling and input validation

## License

This project is licensed under the MIT License - see the `LICENSE` file for details. (Note: A `LICENSE` file is not currently present in the repository. It is recommended to add one.)

# Finance RAG Project

This project implements a Retrieval-Augmented Generation (RAG) system for financial information using semantic search and document retrieval.

## Recent Updates and Fixes

### Version 4.1 (June 2025)
- Fixed `ModuleNotFoundError: No module named 'tools'` by clarifying PyMuPDF installation
- Updated installation instructions to **remove the old `fitz` package** and install only `PyMuPDF`
- Added troubleshooting steps for Python package conflicts

### Version 2.0 (Latest)
- Simplified the query system by removing Phi-2 model dependency
- Fixed token length issues by implementing direct document retrieval
- Improved response formatting for better readability
- Added document summarization with title and content preview

### Previous Issues and Solutions
1. Token Length Error
   - Issue: Input length exceeded model's maximum token limit
   - Solution: Removed complex language model generation in favor of direct document retrieval
   - Benefits: More reliable, faster responses, no token limitations

2. Document Processing
   - Issue: Complex document processing with multiple models
   - Solution: Simplified to use only SentenceTransformer for semantic search
   - Benefits: Reduced complexity, improved stability

## Detailed Setup Instructions

1. **Install Python 3.10.9**
   - Download Python 3.10.9 from [Python's official website](https://www.python.org/downloads/release/python-3109/)
   - For Windows: Download and run the Windows installer (64-bit)
   - Make sure to check "Add Python to PATH" during installation
   - Verify installation by opening a terminal and running:
     ```bash
     python --version
     ```

2. **Create and Activate Virtual Environment**
   - Open a terminal/command prompt
   - Navigate to your project directory
   - Create a new virtual environment:
     ```bash
     # If you have multiple Python versions installed, specify Python 3.10 explicitly:
     # On Windows
     python3.10 -m venv venv
   venv\Scripts\activate

     # On Linux/Mac
     python3.10 -m venv venv
     source venv/bin/activate

     # If you only have Python 3.10 installed, you can use:
     # On Windows
     python -m venv venv
     venv\Scripts\activate

     # On Linux/Mac
     python3 -m venv venv
     source venv/bin/activate
     ```
   - You should see `(venv)` at the start of your command prompt
   - Verify Python version in the virtual environment:
     ```bash
     python --version  # Should show Python 3.10.x
     ```

3. **Upgrade pip and Install Dependencies**
   ```bash
   # Upgrade pip to latest version
   python -m pip install --upgrade pip

   # Install main requirements
   pip install -r requirements.txt

   # Remove any old or conflicting fitz package
   pip uninstall -y fitz

   # Install PyMuPDF (provides the correct fitz module)
   pip install PyMuPDF

   # Install additional required packages
   pip install playwright-stealth
   ```
   - **Note:** Do **not** install the standalone `fitz` package. Only `PyMuPDF` is required for PDF processing.

4. **Install Playwright Browsers**
   ```bash
   # Install Playwright browsers
   playwright install
   ```

5. **Verify Installation**
   ```bash
   # Check if all required packages are installed
   pip list
   ```

6. **Run the Scraper**
   ```bash
   # First, run the scraper to collect financial data
   python scraper.py
   ```
   - This will create the necessary data files in the `data/` directory
   - Wait for the scraping process to complete
   - Check `scraper.log` for any errors

7. **Start the Query System**
   ```bash
   # Run the query interface
   python rag_query.py
   ```

## Troubleshooting Common Issues

1. **ModuleNotFoundError: No module named 'tools'**
   - This error occurs if the wrong `fitz` package is installed.
   - **Solution:**  
     - Uninstall the incorrect package:
       ```bash
       pip uninstall -y fitz
       ```
     - Install the correct package:
       ```bash
       pip install PyMuPDF
       ```
     - Try running your script again.

2. **Multiple Python Versions**
   - If you have multiple Python versions installed:
     - Use `python3.10` explicitly when creating the virtual environment
     - Verify the correct Python version is being used with `python --version`
     - If the wrong version is activated, deactivate and recreate the environment:
       ```bash
       deactivate
       rm -rf venv  # On Linux/Mac
       rmdir /s /q venv  # On Windows
       python3.10 -m venv venv
       ```

3. **Missing Dependencies**
   - If you see "ModuleNotFoundError", make sure:
     - Your virtual environment is activated
     - All packages are installed correctly
     - Try reinstalling the specific package:
       ```bash
       pip install <package-name>
       ```

4. **Playwright Issues**
   - If you encounter browser-related errors:
     ```bash
     playwright install --force
     ```

5. **Data Loading Errors**
   - Ensure the `data/` directory exists
   - Check if `finance_texts.json` and `finance_index.faiss` are present
   - If files are missing, run the scraper again

6. **Memory Issues**
   - If you encounter memory errors, try:
     - Closing other applications
     - Reducing the number of documents processed
     - Using a machine with more RAM

## Project Structure

- `rag_query.py`: Main query interface
- `scraper.py`: Web scraper for financial data
- `data/`: Directory containing:
  - `finance_texts.json`: Processed financial documents
  - `finance_index.faiss`: FAISS index for semantic search
- `requirements.txt`: Project dependencies

## Features

- Semantic search for financial queries
- Document retrieval with relevance scoring
- Formatted responses with document titles and content
- Error handling and logging
- Interactive command-line interface

## Example Queries

You can ask questions about:
- Interest rates and their calculation
- Stock market concepts
- Investment strategies
- Financial portfolios
- Economic indicators
- Banking and finance concepts

## Error Handling

The system includes comprehensive error handling for:
- Missing dependencies
- File not found errors
- Data loading issues
- Query processing errors
- Invalid inputs

## Logging

- All operations are logged to `scraper.log`
- Log level: INFO
- Format: Timestamp - Level - Message

## Dependencies

- sentence-transformers
- faiss-cpu
- numpy
- requests
- beautifulsoup4
- playwright
- PyMuPDF  <!-- Updated: Only PyMuPDF, not fitz -->
- playwright-stealth

## Contributing

Feel free to submit issues and enhancement requests!
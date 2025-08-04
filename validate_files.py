from datetime import datetime
from pathlib import Path

# Validate input files
def validate_files(DIRECTORIES):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Validating input files...")
    file_path = Path(DIRECTORIES['PREPROCESSED_FILE'])
    
    if not file_path.exists():
        error_msg = f"Error: PREPROCESSED_FILE not found at {file_path}"
        print(f"## File Validation\n{error_msg}\n")
        raise FileNotFoundError(error_msg)
        
    print("## File Validation\nInput file found.\n")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Input file validated successfully.")



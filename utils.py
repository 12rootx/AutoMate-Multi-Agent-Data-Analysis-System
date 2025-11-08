from typing import List, Dict, Any
from pathlib import Path
import os
import pandas as pd
import numpy as np
import re

# === Files ===
def discover_files(upload_path: str = None, patterns: List[str] = None) -> List[str]:
    """Discover data files in directory or uploaded files"""
    if patterns is None:
        patterns = ['*.csv', '*.parquet', '*.xlsx', '*.json']

    discovered_files = []
    if upload_path and os.path.exists(upload_path):
        source_dir = Path(upload_path)
        for pattern in patterns:
            files = list(source_dir.rglob(pattern))
            discovered_files.extend([str(f) for f in files if not f.name.startswith('.')])
    return discovered_files


# Load datasets

def load_single_file(file_path: str) -> pd.DataFrame:
    """Load a single file based on extension"""
    file_path = Path(file_path)

    if file_path.suffix.lower() == '.csv':
        return pd.read_csv(file_path, low_memory=False)
    elif file_path.suffix.lower() == '.parquet':
        return pd.read_parquet(file_path)
    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    elif file_path.suffix.lower() == '.json':
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

        
def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Load uploaded file from Streamlit"""
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.parquet'):
        return pd.read_parquet(uploaded_file)
    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.json'):
        return pd.read_json(uploaded_file)
    else:
        raise ValueError(f"Unsupported file format: {uploaded_file.name}")


def load_multiple_files(file_paths: List[str] = None, uploaded_files: List = None) -> Dict[str, pd.DataFrame]:
    """Load multiple files with different schemas"""
    datasets = {}

    if file_paths:
        for file_path in file_paths:
            try:
                df = load_single_file(file_path)
                if df is not None:
                    key = Path(file_path).stem
                    datasets[key] = df
                    #print(f"✅ Loaded {key}: {df.shape}")
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")

    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                df = load_uploaded_file(uploaded_file)
                if df is not None:
                    key = uploaded_file.name.split('.')[0]
                    datasets[key] = df
                    #print(f"✅ Loaded {key}: {df.shape}")
            except Exception as e:
                print(f"❌ Error loading {uploaded_file.name}: {e}")

    return datasets


# === Dataset exploration ===
def analyze_schemas(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    """Analyze schemas of all datasets"""
    schema_analysis = {}

    for name, df in datasets.items():
        schema_analysis[name] = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'sample_values': {}
        }

        for col in df.columns[:3]:  # Limit to first 3 columns
            schema_analysis[name]['sample_values'][col] = df[col].dropna().head(3).tolist()

    return schema_analysis



def extract_python_code(response_text: str) -> str:
    """Extract Python code from LLM response"""
    # Look for code blocks
    code_blocks = re.findall(r'```python(.*?)```', response_text, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    
    # If no code blocks, try to find indented code
    lines = response_text.split('\n')
    python_lines = [line for line in lines if not line.strip().startswith(('#', '"', "'")) and line.strip()]
    return '\n'.join(python_lines)

def execute_generated_code(code: str, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Safely execute the generated Python code"""
    
    # Create a safe execution environment
    safe_globals = {
        'pd': pd,
        'np': np,
        'datasets': datasets,
        '__builtins__': __builtins__
    }
    
        # Add basic dataframe operations to safe environment
    safe_globals.update({
            'merge': pd.merge,
            'concat': pd.concat,
            'DataFrame': pd.DataFrame,
            'Series': pd.Series
        })

        # Execute the code
    exec(code, safe_globals)
        
        # Get the final dataframe
    final_df = safe_globals.get("final_df")
        
    if final_df is not None and isinstance(final_df, pd.DataFrame):
        print(f"✅ Code execution successful! Final shape: {final_df.shape}")
        return final_df
    else:
        raise ValueError("Generated code did not produce a valid DataFrame")
            

def generate_schema_summary(datasets: Dict[str, pd.DataFrame]) -> Dict:
    """Generate comprehensive schema information for code generation"""
    
    schema_summary = {}
    for name, df in datasets.items():
        schema_summary[name] = {
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'shape': (int(df.shape[0]), int(df.shape[1])),
            'sample_data': get_sample_data(df),
            'key_characteristics': get_data_characteristics(df)
        }
    return schema_summary

def get_sample_data(df: pd.DataFrame, n: int = 3) -> Dict:
    """Get sample data for code generation context"""
    samples = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            samples[col] = {
                'type': 'numeric',
                'sample': df[col].dropna().head(n).tolist()
            }
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            samples[col] = {
                'type': 'datetime',
                'sample': df[col].dropna().head(n).dt.strftime('%Y-%m-%d').tolist()
            }
        else:
            samples[col] = {
                'type': 'categorical',
                'sample': df[col].dropna().head(n).astype(str).tolist()
            }
    return samples

def get_data_characteristics(df: pd.DataFrame) -> Dict:
    """Get data characteristics to help with code generation"""
    return {
        'numeric_columns': df.select_dtypes(include=np.number).columns.tolist(),
        'date_columns': df.select_dtypes(include=['datetime']).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'has_missing': df.isnull().any().any(),
        'row_count': len(df)
    }

def fallback_processing(datasets: Dict[str, pd.DataFrame], user_query: str) -> pd.DataFrame:
    """Fallback processing when code generation fails"""
    print("🔄 Using fallback processing...")
    
    if len(datasets) == 1:
        return list(datasets.values())[0]
    
    # Simple heuristic-based fallback
    dfs = list(datasets.values())
    
    # Try to find common columns for joining
    common_cols = set(dfs[0].columns)
    for df in dfs[1:]:
        common_cols = common_cols.intersection(df.columns)
    
    if common_cols:
        # Join on common columns
        result = dfs[0]
        for df in dfs[1:]:
            common = list(common_cols.intersection(df.columns))
            if common:
                result = result.merge(df, on=common[0], how='left', suffixes=('', '_right'))
        return result
    
    return pd.concat(dfs, axis=1)

## Advanced Version with Code Validation
def validate_and_execute_code(code: str, datasets: Dict) -> pd.DataFrame:
    """Validate code before execution"""
    
    # Basic security checks
    forbidden_patterns = [
        r'import\s+os', r'import\s+sys', r'__import__', r'exec\s*\(', r'eval\s*\(',
        r'open\s*\(', r'file\s*\(', r'subprocess', r'os\.', r'sys\.', r'rmdir',
        r'remove', r'delete', r'while True', r'import\s+subprocess'
    ]
    
    for pattern in forbidden_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            raise SecurityError(f"Forbidden pattern detected: {pattern}")
    
    return execute_generated_code(code, datasets)

class SecurityError(Exception):
    pass



# More comprehensive text cleaning
def clean_business_text(text):
    """Fix common text formatting issues in business insights"""
    if not text:
        return text
    
    # Fix smart quotes and special characters
    replacements = {
        '”': '"', '“': '"',
        '’': "'", '‘': "'", 
        '–': '-', '—': '-',
        '…': '...',
        '”': '"', '“': '"'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Fix ALL number+word and word+number combinations
    import re
    # Fix: number followed by word (114,640.26and -> 114,640.26 and)
    text = re.sub(r'([0-9,]+\.?\d*)([a-zA-Z])', r'\1 \2', text)
    # Fix: word followed by number (comparedto” -> compared to ”)
    text = re.sub(r'([a-zA-Z])([0-9,]+\.?\d*)', r'\1 \2', text)
    # Fix: currency amounts stuck to words ($170,188.07in -> $170,188.07 in)
    text = re.sub(r'(\$[0-9,]+\.?\d*)([a-zA-Z])', r'\1 \2', text)
    # Fix: punctuation spacing (health,eauty -> health, eauty)
    text = re.sub(r'([a-zA-Z])([.,!?;:])([a-zA-Z])', r'\1\2 \3', text)
    
    # Fix multiple spaces
    text = re.sub(r' +', ' ', text)
    
    return text.strip()


import zipfile       # <--- Add this line
from PIL import Image # <--- Add this line
import requests
import pandas as pd
from typing import Any, Optional
import io
import json
import pdfplumber
import csv
import base64 # Added for base64 encoding of audio bytes
import time   # Added for exponential backoff
import os
from dotenv import load_dotenv
import urllib
import urllib.parse


#-- NEW HELPER FUNCTION ---to manage API Keys and URL construction dynamically --#
def _get_api_url() -> Optional[str]:
    """Retrieves and constructs the Gemini API URL dynamically."""
    # Ensure environment variables are loaded if not already present
    load_dotenv() 

    api_key = os.getenv("API_KEY")
    model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash") # Use a safe default
    api_url_template = os.getenv("API_URL_TEMPLATE", 
        "https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"
    ) # Use a safe default template

    if not api_key:
        print("Error: API_KEY not found.")
        return None

    return api_url_template.format(MODEL_NAME=model_name, API_KEY=api_key)

# FUNCTION TO DOWNLOAD FILE CONTENT
def download_file_content(url: str) -> Optional[bytes]:
    """
    Downloads raw content (e.g., PDF or CSV bytes) from a URL using synchronous requests.
    This function is called by asyncio.to_thread() in solver.py.
    Returns raw bytes if successful, None otherwise.
    """
    try:
        print(f"-> Downloading static file content from: {url}")
        # Use requests for static file download
        response = requests.get(url, timeout=30)  # increased timeout for robustness
        response.raise_for_status()
        # --- NEW: Smart GitHub Fetcher ---
        if url.endswith("gh-tree.json"):
            try:
                print("    [Info] Detected GitHub Config. Fetching full tree...")
                config = response.json()
                owner, repo, sha = config.get("owner"), config.get("repo"), config.get("sha")
                if owner and repo and sha:
                    gh_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{sha}?recursive=1"
                    gh_res = requests.get(gh_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
                    gh_res.raise_for_status()
                    return gh_res.content # Return the actual file list, not the config
            except Exception as e:
                print(f"    [Warning] GitHub auto-fetch failed: {e}")
        # ---------------------------------
        return response.content
    except Exception as e:
        print(f"Error downloading file {url}: {e}")
        return None

# FUNCTION TO TRANSCRIBE AUDIO CONTENT
def transcribe_audio_content(audio_bytes: bytes, mime_type: str = "audio/ogg") -> Optional[str]:
    """
    Transcribes raw audio content bytes using the Gemini API (multimodal input).
    
    This function implements exponential backoff for handling rate limits/transient errors.

    Args:
        audio_bytes: The raw bytes of the audio file.
        mime_type: The MIME type of the audio file (e.g., "audio/ogg").
        
    Returns:
        The transcribed text string, or None if transcription fails.
    """
    # Getting the URL dynamically
    API_URL_WITH_KEY = _get_api_url()
    if not API_URL_WITH_KEY:
        return None
    
    # Extract key from URL and use Authorization header
    parsed_url = urllib.parse.urlparse(API_URL_WITH_KEY)
    
    # Extract the key from the query parameters
    query_params = urllib.parse.parse_qs(parsed_url.query)
    api_key = query_params.get('key', [None])[0]
    
    if not api_key:
        print("Error: Could not find API key in the URL.")
        return None
        
    # Reconstruct the URL without the key (base URL + path)
    BASE_API_URL = urllib.parse.urlunparse(parsed_url._replace(query=''))
    
    # 3. Implement Exponential Backoff for API Call
    max_retries = 5
    initial_delay = 1
    
    print(f"-> Attempting to transcribe audio content ({len(audio_bytes)} bytes) via Gemini API...")

    # 1. Base64 Encode the Audio Bytes
    encoded_audio = base64.b64encode(audio_bytes).decode('utf-8')

    # 2. Construct the API Payload
    user_prompt = "Transcribe this audio file completely. Do not add any extra commentary or analysis, just the raw text."
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": user_prompt},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": encoded_audio
                        }
                    }
                ]
            }
        ]
    }

    # 3. Implement Exponential Backoff for API Call
    max_retries = 5
    initial_delay = 1 # seconds
    
    for attempt in range(max_retries):
        try:
            # FIX: Send key as query param, NOT as Bearer token header
            headers = {
                'Content-Type': 'application/json'
            }
            # Append key to the URL
            url_with_key = f"{BASE_API_URL}?key={api_key}"

            response = requests.post(url_with_key, headers=headers, data=json.dumps(payload), timeout=30)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

            result = response.json()
            
            # Check for content and extract the transcribed text
            candidate = result.get('candidates', [{}])[0]
            transcribed_text = candidate.get('content', {}).get('parts', [{}])[0].get('text')

            if transcribed_text:
                print("-> Transcription successful.")
                return transcribed_text.strip()
            else:
                # If the API returns a successful status but no text, it might be a safety block or internal issue.
                print(f"    [Warning: API call successful, but no transcription text found in response: {result}]")
                return None

        except requests.exceptions.HTTPError as e:
            # Handle specific HTTP errors (like 429 Rate Limit)
            if response.status_code == 429 and attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                print(f"    [Warning: Rate limit hit (429). Retrying in {delay:.2f}s...]")
                time.sleep(delay)
            else:
                print(f"    [Error: HTTP Error after {attempt+1} attempts: {e}]")
                return None
        
        except requests.exceptions.RequestException as e:
            # Handle other request errors (e.g., connection, timeout)
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                print(f"    [Warning: Request failed: {e}. Retrying in {delay:.2f}s...]")
                time.sleep(delay)
            else:
                print(f"    [Error: Request failed after {attempt+1} attempts: {e}]")
                return None

    return None

# FUNCTION TO PROCESS DATA CONTENT
def process_data(data_content: Any, source_type: str, analysis_plan: str) -> Optional[Any]:
    """
    Processes data (bytes or string) into a structured format.
    Returns:
    - pd.DataFrame for CSV/JSON
    - str for PDF/text/page_content
    - None if processing fails
    This function is called by asyncio.to_thread() in solver.py.
    """
    print(f"-> **** DEBUG DEBUG DEBUG Processing data: Type={source_type}, Plan='{analysis_plan}'")

    #add debug print statements SOURCE TYPE AND CONTENT TYPE
    print(f"    [DEBUG] Source Type: {source_type}")
    print(f"    [DEBUG] Data Content Type: {type(data_content)}")
    if data_content is None:
        return None

    # --- NEW: ZIP Handling ---
    if source_type == "zip" or (isinstance(data_content, bytes) and data_content.startswith(b'PK\x03\x04')):
        try:
            with zipfile.ZipFile(io.BytesIO(data_content)) as z:
                # Find the first useful file (CSV, JSON, LOG, TXT)
                for filename in z.namelist():
                    if filename.endswith(('.csv', '.json', '.jsonl', '.txt', '.log', '.md')):
                        print(f"    [Info] Extracting {filename} from zip")
                        with z.open(filename) as f:
                            # Recursively process the extracted file
                            ext = filename.split('.')[-1]
                            if ext in ['log', 'jsonl']: ext = 'json'
                            return process_data(f.read(), ext, analysis_plan)
        except Exception: pass

    # --- NEW: IMAGE Handling (Heatmap Fix) ---
    if source_type in ["png", "image"] and isinstance(data_content, bytes):
        try:
            img = Image.open(io.BytesIO(data_content))
            # Get colors. maxcolors must be > unique colors in image
            colors = img.getcolors(maxcolors=2000000) 
            if colors:
                # Sort by count (descending)
                colors.sort(key=lambda x: x[0], reverse=True)
                most_freq_count, most_freq_color = colors[0]
                
                # Handle RGB vs RGBA
                r, g, b = most_freq_color[:3]
                hex_color = '#{:02x}{:02x}{:02x}'.format(r, g, b)
                
                print(f"    [Success] Analyzed Image. Most frequent color: {hex_color}")
                return hex_color
        except Exception as e:
            print(f"    [Error] Image analysis failed: {e}")

    # --- Existing PDF Check ---
    if source_type == "pdf":
        try:
            with pdfplumber.open(io.BytesIO(data_content)) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            print("    [Success: Extracted text from PDF]")
            return text
        except Exception as e:
            print(f"    [Error: Failed to extract PDF text: {e}]")
            return None

    elif source_type in ["csv", "json"]:
        try:
            # Check if content is bytes (from download) or string (from scrape/error)
            if isinstance(data_content, bytes):
                # Prepare data stream for Pandas
                data_io = io.BytesIO(data_content)
                content_str = data_content.decode('utf-8', errors='ignore') # <--- Added this
            else:
                # Use string data as input stream
                content_str = str(data_content) # <--- Added this
                data_io = io.StringIO(content_str)

            if source_type == "csv":
                
                # --- START ROBUST CSV PARSING (FIXED) ---
                df = None
                is_headerless_numeric = False
                
                # Attempt 1: Check for Headerless Numeric Data (Single Column of Numbers)
                try:
                    data_io.seek(0)
                    # Peek at the first few rows with no header. Use 'python' engine for robustness.
                    df_temp = pd.read_csv(data_io, header=None, engine='python', nrows=5) 
                    
                    # Heuristic Check: Check if it's a single column AND if the column is entirely numeric
                    if len(df_temp.columns) == 1:
                        # Convert to numeric, errors='coerce' turns non-numbers into NaN
                        numeric_col = pd.to_numeric(df_temp.iloc[:, 0], errors='coerce')
                        
                        # If all values in the peeked column are valid numbers, assume headerless numeric data
                        if numeric_col.notna().all():
                            data_io.seek(0) # Reset stream and read the full file again
                            df = pd.read_csv(data_io, header=None, engine='python')
                            
                            # Rename the single column for easy LLM access
                            df = df.rename(columns={df.columns[0]: 'value'})
                            
                            # Final conversion and NaN cleanup
                            df['value'] = pd.to_numeric(df['value'], errors='coerce')
                            is_headerless_numeric = True
                            print("    [Info: Detected and loaded as headerless numeric CSV. Column: 'value']")
                        
                except Exception as e:
                    print(f"    [Warning: Failed headerless attempt: {e}]")
                    
                
                # Attempt 2: Standard CSV Processing (if headerless failed or was not appropriate)
                if not is_headerless_numeric:
                    data_io.seek(0) # Reset stream position
                    # Load with default header=0 (uses first row as header)
                    # This handles standard headers and multi-column data robustly
                    df = pd.read_csv(data_io, engine='python')
                    print("    [Info: Loaded as standard CSV (header inferred).]")
                    
                    # Universal step: Convert columns to numeric where possible
                    df = df.apply(pd.to_numeric, errors='ignore')

                df = df.dropna(how='all') # Remove entirely empty rows
                # --- END ROBUST CSV PARSING (FIXED) ---
                
            else:
                # JSON / JSONL
                try:
                    obj = json.loads(content_str)
                    
                    # --- FIX: Flatten GitHub Tree ---
                    # If the JSON has a "tree" list (standard GitHub API), use that as the data
                    if isinstance(obj, dict) and "tree" in obj and isinstance(obj["tree"], list):
                        print("    [Info] Flattened GitHub Tree JSON to expose file list.")
                        df = pd.DataFrame(obj["tree"])
                    # --------------------------------
                    elif isinstance(obj, list):
                        df = pd.DataFrame(obj)
                    else:
                        df = pd.DataFrame([obj])
                except:
                    # JSONL fallback
                    obj = [json.loads(line) for line in content_str.splitlines() if line.strip()]
                    df = pd.DataFrame(obj)

            print(f"    [Success: Data loaded into DataFrame. Columns: {list(df.columns)}]")
            return df
        except Exception as e:
            print(f"    [Error: Failed to load data into DataFrame: {e}]")
            # If pandas/cleaning fails, return the raw text for the LLM to interpret
            return str(data_content)

    # --- 5. SQL Handling ---
    if source_type == "sql":
        try:
            return data_content.decode('utf-8', errors='ignore')
        except:
            return str(data_content)

    elif source_type in ["text", "page_content"]:
        print("    [Success: Data is page text. Ready for LLM re-prompt.]")
        return str(data_content)

    # Fallback: return as string
    return str(data_content)

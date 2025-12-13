import os
import time
import json
import asyncio
import httpx # Required for async HTTP requests (submission and helper)
import requests # Still needed for synchronous download_file_content (which is wrapped)
from playwright.async_api import async_playwright
from google import genai
from google.genai.errors import APIError
import pandas as pd
import urllib.parse 
from typing import Any, Dict, Optional, Tuple, Union
import re # Added for parsing the analysis plan
from dotenv import load_dotenv
from data_tools import download_file_content, process_data, transcribe_audio_content

# --- INITIALIZATION AND CONFIGURATION FIXES ---

# 1. NEW HELPER: Universal URL Resolver (Fix for Issue 3)
def resolve_relative_url(base_url: str, path: str) -> str:
    """Converts a relative path to an absolute URL using the base URL."""
    # Ensure base_url ends with '/' for correct resolution
    base = urllib.parse.urlunparse(urllib.parse.urlparse(base_url)[:2] + ('/', '', '', ''))
    return urllib.parse.urljoin(base, path)

# 2. NEW HELPER: Client Getter (Fix for Issue 1)
_gemini_client = None
def get_gemini_client() -> Optional[genai.Client]:
    """Initializes and returns the Gemini client (Singleton pattern)."""
    global _gemini_client
    if _gemini_client is None:
        # Load environment variables just before client initialization
        load_dotenv() 
        api_key = os.getenv("API_KEY")

        if api_key:
            try:
                _gemini_client = genai.Client(api_key=api_key)
                print("Gemini client initialized successfully.")
            except Exception as e:
                print(f"Error initializing Gemini client: {e}")
                _gemini_client = None
        else:
            print("WARNING: API_KEY not found. Gemini client not initialized. LLM steps will fail.")
    return _gemini_client
# -----------------------------------------------

# Load environment variables to ensure API_KEY is available for client initialization
load_dotenv()
API_KEY = os.getenv("API_KEY")

# --- INITIALIZE GEMINI CLIENT ---
if API_KEY:
    try:
        # The client object that was previously undefined
        client = genai.Client(api_key=API_KEY)
        print("Gemini client initialized successfully.")
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        client = None
else:
    print("WARNING: API_KEY not found. Gemini client not initialized. LLM steps will fail.")
    client = None
# ----------------------------------


# Assuming data_tools is in the same directory and contains the necessary imports
from data_tools import download_file_content, process_data, transcribe_audio_content


RE_PATTERN = r'Post your answer to\s+["\']?([^\s"\']+)["\']?'

MAX_TIME_SECONDS = 180 # 3 minutes total for all submissions for this URL
MAX_ATTEMPTS = 3

# --- NEW HELPER: FIND DATA URLS ON PAGE ---
async def find_data_urls_on_page(page, quiz_url: str) -> Dict[str, Optional[str]]:
    """
    Uses Playwright to scrape the page for common data file links (CSV, Audio).
    """
    domain_root = urllib.parse.urlunparse(urllib.parse.urlparse(quiz_url)[:2] + ('/', '', '', ''))

    results = await page.evaluate('''() => {
        const audioExtensions = ['.mp3', '.wav', '.ogg', '.opus', '.flac', '.aac'];
        const dataExtensions = ['.csv', '.json', '.pdf', '.txt', '.zip', '.log'];
        let audio_url = null;
        let csv_url = null;
        
        // Check <a> links
        document.querySelectorAll('a').forEach(a => {
            const href = a.getAttribute('href') || '';
            if (!audio_url && audioExtensions.some(ext => href.toLowerCase().endsWith(ext))) {
                audio_url = href;
            }
            if (!csv_url && dataExtensions.some(ext => href.toLowerCase().endsWith(ext))) {
                csv_url = href;
            }
        });
        
        // Check <audio>/<source> tags
        if (!audio_url) {
            document.querySelectorAll('audio, source').forEach(el => {
                const src = el.getAttribute('src') || '';
                 if (audioExtensions.some(ext => src.toLowerCase().endsWith(src))) {
                    audio_url = audio_url || src; // prioritize the first one found
                }
            });
        }
        
        return { audio_url, csv_url };
    }''')
    
    # Resolve relative paths
    #audio_url = urllib.parse.urljoin(domain_root, results.get('audio_url')) if results.get('audio_url') else None
    #csv_url = urllib.parse.urljoin(domain_root, results.get('csv_url')) if results.get('csv_url') else None
    
    audio_url = resolve_relative_url(quiz_url, results.get('audio_url')) if results.get('audio_url') else None
    csv_url = resolve_relative_url(quiz_url, results.get('csv_url')) if results.get('csv_url') else None
# ...

    # Simple deduplication/prioritization
    if audio_url == csv_url:
        csv_url = None # Prioritize as audio if the URL is an audio file.

    return {
        "audio_url": audio_url,
        "csv_url": csv_url
    }

# --- PANDAS EXECUTION HELPER (CRITICALLY FIXED) ---
def execute_pandas_analysis(df: pd.DataFrame, plan: str) -> Optional[Union[int, float]]:
    """
    Attempts to parse a simple analysis plan (Filter and Sum) and execute it using Pandas.
    Returns the final calculated value (sum) if successful, otherwise None.
    
    CRITICAL FIX: Uses explicit boolean indexing (if/elif) and refactored operator parsing 
    to bypass the unstable df.query() numexpr engine and the "Unsupported operator" error.
    """
    plan_lower = plan.lower()

    # --- NEW: HANDLE JSON FORMATTING TASKS DIRECTLY ---
    # Only run this if it's a PURE formatting task (no counting/summing requested)
    if "json" in plan_lower and ("convert" in plan_lower or "format" in plan_lower) and "count" not in plan_lower and "sum" not in plan_lower:
        try:
            print("-> Pandas Analysis: Detected JSON formatting task. Executing via Pandas...")
            # 1. Normalize Columns (Snake Case)
            df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]
            
            # 2. Rename specific columns to match requirements if needed
            # Common variations: 'id', 'name', 'joined', 'value'
            # If columns are like 'id', 'name', 'joined_date', 'value', rename 'joined_date' -> 'joined'
            rename_map = {}
            for col in df.columns:
                if 'joined' in col: rename_map[col] = 'joined'
                if 'date' in col and 'joined' not in col: rename_map[col] = 'joined'
            if rename_map:
                df = df.rename(columns=rename_map)

            # 3. Standardize Dates (ISO 8601 YYYY-MM-DD)
            if 'joined' in df.columns:
                # Try converting with dayfirst=False (Month-First default for US/International mix)
                # If that fails or looks wrong, the loop logic in solver.py (re-attempt) won't help here 
                # unless we expose it. For now, standard pd.to_datetime usually handles 'YYYY-MM-DD' and 'DD Month YYYY' well.
                # The ambiguous '02/01/24' is the killer. 
                # Strategy: Use mixed inference. Set dayfirst=True to match "1 Feb 2024" style context.
                df['joined'] = pd.to_datetime(df['joined'], dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%d')

            # 4. Ensure Integers
            if 'value' in df.columns:
                df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(0).astype(int)
            if 'id' in df.columns:
                df['id'] = pd.to_numeric(df['id'], errors='coerce').fillna(0).astype(int)
                # 5. Sort by ID
                df = df.sort_values(by='id')

            # 6. Export to JSON String (list of objects)
            json_output = df.to_json(orient='records', indent=2)
            print(f"-> Pandas Analysis SUCCESS: Generated JSON length {len(json_output)}")
            return json_output

        except Exception as e:
            print(f"-> Pandas Analysis JSON Error: {e}")
            return None
    
    # 1. Check for necessary keywords and the expected column 'value'
    if "filter" in plan_lower and "sum" in plan_lower and "value" in df.columns.tolist():
        try:
            op_symbol = None
            val_str = None
            
            # 2. Try Symbolic Match (e.g., >= 53122)
            match = re.search(r"(>=|<=|==|!=|>|<)\s*(\d+)", plan_lower)
            if match:
                op_symbol, val_str = match.groups()
            else:
                # 3. Try Plain Text Match (e.g., greater than or equal to 53122)
                # Regex for plain text: (operator phrase) followed by (value)
                op_match = re.search(r"(greater\s+than\s+or\s+equal\s+to|greater\s+than|less\s+than\s+or\s+equal\s+to|less\s+than|equal\s+to)\s*(\d+)", plan_lower)
                
                if op_match:
                    op_text, val_str = op_match.groups()
                    
                    # Convert plain text to symbolic operator
                    if "greater than or equal to" in op_text: op_symbol = ">="
                    elif "greater than" in op_text: op_symbol = ">"
                    elif "less than or equal to" in op_text: op_symbol = "<="
                    elif "less than" in op_text: op_symbol = "<"
                    elif "equal to" in op_text: op_symbol = "=="
            
            # 4. Execution based on derived symbol
            if op_symbol and val_str:
                val = int(val_str)
                col = 'value' 
                
                # --- CRITICAL FIX: Use Boolean Indexing (Bypasses numexpr error) ---
                # This ensures stability regardless of the execution environment's numexpr/pandas setup.
                query_string = f"{col} {op_symbol} {val} (using boolean indexing)"
                
                if op_symbol == '>=':
                    filtered_df = df[df[col] >= val]
                elif op_symbol == '>':
                    filtered_df = df[df[col] > val]
                elif op_symbol == '<=':
                    filtered_df = df[df[col] <= val]
                elif op_symbol == '<':
                    filtered_df = df[df[col] < val]
                elif op_symbol == '==' or op_symbol == '=':
                    filtered_df = df[df[col] == val]
                else:
                    print(f"-> Pandas Analysis Runtime Error: Unsupported operator {op_symbol}")
                    return None
                # --- END CRITICAL FIX ---
                
                if not filtered_df.empty:
                    result = filtered_df[col].sum()
                    result = int(result) if result == int(result) else result
                    
                    print(f"-> Pandas Analysis SUCCESS: Executed query '{query_string}'. Rows matched: {len(filtered_df)}. Final Sum: {result}")
                    return result
                else:
                    print(f"-> Pandas Analysis Warning: Filter query '{query_string}' resulted in 0 rows. Sum is 0.")
                    return 0
            else:
                print("-> Pandas Analysis Warning: Could not parse filter and sum logic from plan.")
                
        except Exception as e:
            print(f"-> Pandas Analysis Runtime Error: {e}")
            return None
    
    return None

# --- ASYNC HELPER FOR SUBMISSION (Uses httpx) ---
async def fetch_with_retry(url: str, method: str = 'POST', data: Dict[str, Any] = None, headers: Dict[str, str] = None, max_retries: int = 5):
    """Handles HTTP requests with exponential backoff using httpx (async)."""
    # Use a new client session for each call for simplicity, though a shared one is more efficient
    client_session = httpx.AsyncClient(timeout=60.0) 
    # Ensure Content-Type is set for POST requests
    headers = headers or {}
    if method == 'POST':
        headers['Content-Type'] = 'application/json'

    for i in range(max_retries):
        try:
            if method == 'POST':
                response = await client_session.post(url, json=data, headers=headers)
            else: # Assuming GET
                response = await client_session.get(url, headers=headers)
            response.raise_for_status() 
            return response.json()
        except httpx.HTTPStatusError as e:
            if response.status_code == 429 or 500 <= response.status_code < 600:
                print(f"Transient error: {response.status_code}. Retrying in {2**i}s...")
                await asyncio.sleep(2**i)
                continue
            raise e
        except httpx.RequestError as e:
            print(f"Request error: {e}. Retrying in {2**i}s...")
            await asyncio.sleep(2**i)
            continue
    raise Exception(f"Failed to fetch {url} after {max_retries} attempts.")

async def scrape_dynamic_page(page, url: str) -> str:
    """Navigates to a new URL and scrapes the body text content."""
    print(f"    [Dynamic Fetch] Navigating to new URL: {url}")
    # Use 'networkidle' to ensure all necessary content is loaded
    await page.goto(url, wait_until="networkidle", timeout=30000) 
    return await page.inner_text("body")

def solve_quiz(email: str, secret: str, initial_url: str):
    """
    Synchronous wrapper to start the asynchronous loop. Handles the common
    "Event loop is already running" error when run inside FastAPI/Uvicorn.
    """
    print(f"Starting quiz loop for: {initial_url}")
    # CRITICAL FIX: Ensure client is initialized here (Fix for Issue 1)
    client = get_gemini_client()
    
    print(f"Starting quiz loop for: {initial_url}")
    if client is None:
        print("FATAL ERROR: Gemini client is not initialized. Cannot proceed with LLM analysis.")
        return
        
    try:
        # Run in a new event loop
        asyncio.run(quiz_loop(email, secret, initial_url, time.time(), client))
    except RuntimeError as e:
        if "Event loop is already running" in str(e):
             print("ASYNCIO RUNTIME ERROR: Event loop already running (expected in FastAPI). Running as detached task.")
             # If the loop is running (Uvicorn), create a detached task
             asyncio.create_task(quiz_loop(email, secret, initial_url, time.time()))
        else:
            raise e
    except Exception as e:
        print(f"FATAL QUIZ LOOP ERROR: {e}")

async def quiz_loop(email: str, secret: str, initial_url: str, start_time: float, client: genai.Client):
    """
    The main asynchronous function, using a non-recursive while loop to solve 
    the quiz chain.
    """
    
    current_url = initial_url
    
    while current_url:
        start_time = time.time() # <--- RESETS TIMER FOR EVERY NEW QUESTION
        time.sleep(12)
        quiz_url = current_url
        print(f"\n--- Processing Quiz URL: {quiz_url} ---")
        
        elapsed_time = time.time() - start_time
        if elapsed_time > MAX_TIME_SECONDS:
            print("TIME LIMIT EXCEEDED BEFORE STARTING. Abandoning quiz chain.")
            return

        browser = None
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()

                # --- NEW DEBUG STATEMENT HERE ---
                print(f"    [DEBUG] Quiz run started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
                # ---------------------------------
                
                # --- 1. VISIT AND EXTRACT QUIZ TASK ---
                await page.goto(quiz_url, wait_until="networkidle", timeout=60000) 
                quiz_content = await page.inner_text("body")
                
                # --- 1.2 EXTRACT SUBMISSION URL ---
                submit_url_match = await page.evaluate('''() => {
                    const bodyText = document.body.innerText;
                    
                    // Fixed: Converted Python raw string to correctly escaped JS regex literal
                    let match = bodyText.match(/Post your answer to\\s+["']?([^\\s"']+)["']?/);
                    if (match && match[1]) return match[1];
                    
                    // Fixed: Double-escaping backslashes for Python string compliance
                    match = bodyText.match(/(https:\\/\\/[^"\\s]*submit[^"\\s]*)/i);
                    if (match && match[1]) return match[1];
                    // Fixed: Double-escaping backslashes for Python string compliance
                    match = bodyText.match(/(\\/\\w*submit\\w*)/i);
                    if (match && match[1]) return match[1];
                    
                    return null;
                }''')

                if not submit_url_match:
                    print("WARNING: URL not found. Forcing default submit URL.")
                    submit_url_match = "https://tds-llm-analysis.s-anand.net/submit"

                # Resolve relative URLs
                if submit_url_match.startswith('/'):
                    # Ensure base_url ends with '/' for correct resolution
                    base_url = urllib.parse.urlunparse(urllib.parse.urlparse(quiz_url)[:2] + ('/', '', '', ''))
                    #submit_url_match = urllib.parse.urljoin(base_url, submit_url_match)
                    submit_url_match = resolve_relative_url(quiz_url, submit_url_match)

                print(f"Submission Target: {submit_url_match}")
                
                # --- START ATTEMPTS LOOP ---
                attempts = 0
                previous_error = None
                submission_result = {}
                while attempts < MAX_ATTEMPTS: 
                    attempts += 1
                    
                    if time.time() - start_time > MAX_TIME_SECONDS:
                        print(f"TIME LIMIT EXCEEDED ({MAX_TIME_SECONDS}s). Abandoning submission.")
                        await browser.close()
                        return
                    
                    final_answer = None
                    llm_plan = None 

                    try:
                        # --- NEW PRE-PROCESSING STEP: FIND DATA URLS ---
                        found_urls = await find_data_urls_on_page(page, quiz_url)
                        csv_url_found = found_urls.get('csv_url')
                        audio_url_found = found_urls.get('audio_url')
                        
                        print(f"-> Found CSV URL: {csv_url_found}")
                        print(f"-> Found Audio URL: {audio_url_found}")
                        
                        # Step 1: Transcribe audio if found
                        transcript_text = ""
                        if audio_url_found:
                            print(f"-> Transcribing audio from: {audio_url_found}")
                            
                            # download_file_content is synchronous, wrap it
                            audio_bytes = await asyncio.to_thread(download_file_content, audio_url_found)
                            
                            if audio_bytes:
                                # transcribe_audio_content is synchronous, wrap it
                                ## NOTE: I am keeping your existing transcription call as you confirmed it was working
                                transcript = await asyncio.to_thread(transcribe_audio_content, audio_bytes) 
                                if transcript:
                                    print("-> Transcription successful. Incorporating into prompt.")
                                    print(f"DEBUG TRANSCRIPTION: {transcript}")
                                    transcript_text = "\n\n--- AUDIO TRANSCRIPT ---\n" + transcript + "\n------------------------"
                                
                        # --- 2. LLM ANALYSIS (Plan Generation) ---
                        
                        ## CRITICAL FIX: Prompt refinement to ensure direct answer scraping (Q1 fix)
                        plan_prompt = f"""
                        You are a planning agent.
                        Analyze the quiz content and the audio transcript (if present) to determine the required analysis.
                        
                        The URL also might have audio files embedded. The audio file can have a .opus extension. The file should be preprocessed to convert in a way that it can be transcribed.
                        The transcription text should then be analyzed to extract the final answer. The text can either directly contain the answer or contain data that needs to be further processed to get the final answer.
                        When there are instructions around sum check properly if it is total sum of the individual values of the count of the instances. 
                        
                        1. Identify the final answer type (number, string, etc.).
                        2. Identify the definitive data source.
                        3. The final data source URL should be either the CSV link: '{csv_url_found or 'NONE'}' or the literal string 'PAGE_CONTENT' if the instructions are in the audio transcript or embedded text.
                        **CRITICAL SCRAPING RULE:** If the answer is directly visible in the QUIZ CONTENT or TRANSCRIPT, or if the task is to scrape a new URL for a final text, the 'answer' field MUST be the **exact text extracted**. For the very first question, the answer is the text on the page, even if it looks like a placeholder (e.g., 'anything you want'). If an answer is NOT CALCULATED, use the string "NOT_CALCULATED".
                        
                        **CRITICAL EXTRACTION RULE:** If the audio transcript or quiz content mentions a 'cutoff value' or 'threshold', you MUST find the exact number from the context (e.g., '53122') and use that number in the 'analysis_plan'. Do NOT use a placeholder like 'X' or 'the cutoff value provided'.
                        
                        **IMPORTANT HINT FOR CSV:** If the data source is a headerless CSV (as indicated by the transcription, which implies a single column of numbers), the data processor will rename the first column to 'value'. Ensure your 'analysis_plan' uses the column name 'value' for all calculations (e.g., 'Filter column value >= 53122, then SUM column value').
                        
                        **CRITICAL SCRAPE HINT:** If the quiz content instructs you to navigate to a new URL (e.g., /demo-scrape-data), your `analysis_plan` MUST mention that URL explicitly, and the `data_source_url` should reflect that URL.
                        
                        4. OUTPUT ONLY a clean JSON object with this structure:
                            - `answer`: The final answer, or "NOT_CALCULATED" if data processing is required.
                            - `data_source_url`: URL of external data (e.g., CSV link) or "PAGE_CONTENT" if data is embedded in the quiz text/transcript, or "NONE".
                            - `source_type`: Type of data ('pdf', 'csv', 'json', 'text', 'page_content', 'NONE').
                            - `analysis_plan`: Step-by-step instructions for a data analyst (e.g., "Load CSV, filter by column X > 14687, then SUM column Y.").
                        QUIZ CONTENT:
                        ---
                        {quiz_content}
                        ---
                        {transcript_text}
                        
                        USER EMAIL: {email}  <--- ADD THIS LINE
                        
                        PREVIOUS ERROR (IF ANY):
                        ---
                        {previous_error}
                        ---
                        """
                        
                        # FIX: WRAP BLOCKING SDK CALL IN ASYNCIO
                        llm_response = await asyncio.to_thread(
                            client.models.generate_content,
                            model='gemini-2.5-flash', 
                            contents=plan_prompt
                        )
                        plan_json_str = llm_response.text.strip().replace('```json', '').replace('```', '')
                        llm_plan = json.loads(plan_json_str)
                        
                        data_source_url = llm_plan.get("data_source_url", "NONE")
                        source_type = llm_plan.get("source_type", "NONE").lower()
                        analysis_plan = llm_plan.get("analysis_plan", "")
                        final_answer = llm_plan.get("answer")
                        
                        # --- CRITICAL SOURCE RESOLUTION LOGIC ---
                        
                        # Check 1: Force source URL if a new URL is mentioned in the plan
                        # Look for common URL patterns or relative paths in the analysis plan
                        url_match = re.search(r"(\/[^\s\?]+\?[\w&=@\.-]+)", analysis_plan) # Catches relative URLs like /demo-scrape-data?email=...
                        
                        if url_match:
                            extracted_url = url_match.group(1).strip('.') # strip trailing dots if any
                            
                            # --- FIX: Ignore Template URLs ---
                            if '{' in extracted_url or '}' in extracted_url:
                                print(f"-> Ignoring template URL: {extracted_url}")
                                extracted_url = "" 
                            # ---------------------------------

                            if '{' in extracted_url or '}' in extracted_url:
                                print(f"-> Ignoring template URL: {extracted_url}")
                                extracted_url = "" # Reset so we don't use it

                            if extracted_url: # Only proceed if it's still valid
                                # Resolve the relative URL against the current quiz URL
                                base_url = urllib.parse.urlunparse(urllib.parse.urlparse(quiz_url)[:2] + ('/', '', '', ''))
                                #resolved_url = urllib.parse.urljoin(base_url, extracted_url)
                                resolved_url = resolve_relative_url(quiz_url, extracted_url)
    
                                if resolved_url.lower() != quiz_url.lower():
                                    print(f"-> Source override: Plan mentions new data URL: {resolved_url}. Overriding data_source_url.")
                                    data_source_url = resolved_url
                                    # Ensure we don't accidentally treat this as a static CSV/PDF unless the extension is present
                                    if not resolved_url.lower().endswith(('.csv', '.pdf', '.json')):
                                        source_type = "text"
                                # else: the resolved URL is the current quiz URL, so proceed normally (PAGE_CONTENT)


                        # 2. Check if the LLM's plan requires fetching a static external data file (CSV, PDF, JSON).
                        is_external_data_required = (
                            data_source_url.lower().endswith(".csv") or 
                            data_source_url.lower().endswith(".pdf") or 
                            data_source_url.lower().endswith(".json")
                        )
                        
                        # 3. If audio was transcribed, we only force PAGE_CONTENT if the instructions *don't* point to an external file.
                        if transcript_text and not is_external_data_required:
                             # Keep the LLM's suggested data_source_url which might be the one we just extracted
                             if data_source_url.lower() in ("page_content", "none"):
                                print("-> Source override: Audio transcribed, forcing source to PAGE_CONTENT (since no external data was required).")
                                data_source_url = "PAGE_CONTENT"
                                source_type = "text"
                             
                        # 4. If the LLM identified a CSV/PDF but we already found one, prioritize the found one
                        elif csv_url_found and data_source_url.lower() != csv_url_found.lower() and data_source_url.lower() == "none":
                            print(f"-> Source override: Using directly found CSV URL: {csv_url_found}")
                            data_source_url = csv_url_found
                            source_type = "csv"

                        print(f"Attempt {attempts} | Plan: {analysis_plan} | Source: {data_source_url}")
                        
                        # --- EXECUTE DATA SOURCING/PROCESSING ---
                        if final_answer == "NOT_CALCULATED" or data_source_url not in ("NONE", "PAGE_CONTENT"):
                            data_content = None
                            
                            # --- ROBUST DATA FETCHING LOGIC ---
                            is_static_file = (
                                data_source_url.lower().endswith(".csv") or 
                                data_source_url.lower().endswith(".pdf") or 
                                data_source_url.lower().endswith(".json") or 
                                data_source_url.lower().endswith(".zip") or 
                                data_source_url.lower().endswith(".png") or
                                data_source_url.lower().endswith(".opus") or
                                data_source_url.lower().endswith(".sql")
                            )

                            if data_source_url.lower().endswith(".csv"):
                                print(f"    [FINALDEBUG] Source Type: csv")
                                source_type = "csv"
                            elif data_source_url.lower().endswith(".png"):
                                print(f"    [FINALDEBUG] Source Type: png")
                                source_type = "png"
                            elif data_source_url.lower().endswith(".pdf"):
                                print(f"    [FINALDEBUG] Source Type: pdf")
                                source_type = "pdf"
                            elif data_source_url.lower().endswith(".json"):
                                print(f"    [FINALDEBUG] Source Type: json")
                                source_type = "json"
                            elif data_source_url.lower().endswith(".sql"):
                                print(f"    [FINALDEBUG] Source Type: sql")
                                source_type = "sql"

                            is_static_file = is_static_file and not data_source_url.lower().endswith(('.mp3', '.wav', '.ogg', '.opus', '.flac', '.aac'))
                            print(f"    [FINALDEBUG] Is Static File: {is_static_file}")
                            
                            # Resolve absolute URL for fetching
                            source_url_resolved = data_source_url
                            if data_source_url.startswith('/'):
                                base_url = urllib.parse.urlunparse(urllib.parse.urlparse(quiz_url)[:2] + ('/', '', '', ''))
                                source_url_resolved = urllib.parse.urljoin(base_url, data_source_url)
                                print(f"    [FINALDEBUG] Resolved Source URL: {source_url_resolved}")

                            if is_static_file:
                                print(f"    [FINALDEBUG] Static Fetch (Requests) from: {source_url_resolved}")
                                # Download file synchronously
                                data_content = await asyncio.to_thread(download_file_content, source_url_resolved)
                            
                            elif data_source_url not in ("PAGE_CONTENT", "NONE"):
                                # This handles the scraping task (e.g., /demo-scrape-data?email=...)
                                print(f"    [FINALDEBUG] Dynamic Fetch (Navigate to New URL: {source_url_resolved})")
                                
                                # CRITICAL FIX: The previous version was navigating to the new URL, 
                                # but subsequent code assumed it was still on the old page's context.
                                # To get the data from the NEW page, we must navigate and scrape.
                                #await page.goto(source_url_resolved, wait_until="networkidle", timeout=30000)
                                #data_content = await page.inner_text("body")
                                #source_type = "text" 

                                print(f"    [FINALDEBUG] Dynamic Fetch (Playwright) from: {source_url_resolved}")
                                data_content = await scrape_dynamic_page(page, source_url_resolved)
                                source_type = "text"
                                
                            elif data_source_url == "PAGE_CONTENT" or data_source_url == "NONE":
                                print(f"    [FINALDEBUG] Dynamic Fetch (Current Page Content + Transcript)")
                                # Use quiz_content (original page content) + transcript
                                data_content = quiz_content + transcript_text
                                source_type = "text"
                                
                            # --- END ROBUST DATA FETCHING LOGIC ---
                            
                            if data_content is not None:
                                # Process data (e.g., PDF to text, CSV to DataFrame)
                                extracted_data = await asyncio.to_thread(process_data, data_content, source_type, analysis_plan)
                                
                                # --- NEW: PANDAS CALCULATION PRE-EMPTION ---
                                pandas_result = None
                                if isinstance(extracted_data, pd.DataFrame):
                                    pandas_result = execute_pandas_analysis(extracted_data, analysis_plan)

                                if pandas_result is not None:
                                    final_answer = pandas_result
                                    print(f"-> Pandas calculated final answer: {str(final_answer)[:100]}... (Bypassing LLM)")

                                elif extracted_data is not None:
                                    # Fallback to LLM Re-prompt if not a DataFrame or Pandas execution failed/not applicable
                                    
                                    # Prepare data for LLM re-prompt
                                    if isinstance(extracted_data, pd.DataFrame):
                                        print(f"    [Info] Converting DataFrame to CSV for LLM analysis...")
                                        # Use to_csv to prevent formatting hallucinations and include ALL rows
                                        data_for_llm = extracted_data.to_csv(index=False)
                                    elif isinstance(extracted_data, bytes):
                                         # Safely decode bytes for LLM
                                         data_for_llm = extracted_data[:4000].decode('latin-1', errors='replace')
                                    elif isinstance(extracted_data, str):
                                        data_for_llm = extracted_data[:4000] 
                                    else:
                                        data_for_llm = str(extracted_data)

                                    print("-> Re-prompting LLM for Final Analysis (Pandas not used/failed)...")
                                    
                                    # --- 3. RE-PROMPT LLM FOR FINAL ANALYSIS (SIMPLE TEXT OUTPUT) ---
                                    analysis_prompt = f"""
                                    The original quiz instruction was: "{analysis_plan}".
                                    User Email: {email}
                                    
                                    DATA CONTEXT:
                                    {data_for_llm}
                                    
                                    CRITICAL INSTRUCTIONS:
                                    1. Analyze the data to fulfill: "{analysis_plan}".
                                    2. Output strictly valid JSON if the task asks for JSON. No markdown formatting.
                                    3. Dates MUST be in YYYY-MM-DD format.
                                    4. HANDLE AMBIGUITY: If a date is "02/01/24", it could be Feb 1st or Jan 2nd. 
                                       Look at the other rows to infer the format (DD/MM/YY vs MM/DD/YY).
                                       If previous attempts failed, switch your interpretation.
                                       Normalize ALL dates strictly to YYYY-MM-DD.
                                    5. OUTPUT ONLY the final result.
                                    """

                                    # FIX: WRAP BLOCKING SDK CALL IN ASYNCIO
                                    analysis_response = await asyncio.to_thread(
                                        client.models.generate_content,
                                        model='gemini-2.5-flash', 
                                        contents=analysis_prompt
                                    )
                                    # The only fix here is adding .text to handle potential None return from llm_response if the API failed, 
                                    # preventing the 'NoneType' object has no attribute 'strip' error from the first quiz attempt.
                                    if analysis_response.text:
                                        # Clean up markdown formatting (```json ... ```)
                                        raw = analysis_response.text.strip()
                                        if "```" in raw:
                                            raw = raw.replace("```json", "").replace("```", "")
                                        # Also remove raw "json" if it appears at the start
                                        if raw.startswith("json"):
                                            raw = raw[4:]
                                        final_answer = raw.strip()
                                    else:
                                        final_answer = None
                                else:
                                    print("ERROR: Data processing/extraction failed.")
                                    continue
                                    
                        # Submission Logic 
                        if final_answer is None or final_answer == "NOT_CALCULATED" or len(str(final_answer).strip()) == 0:
                            print("ERROR: Answer could not be determined for submission.")
                            continue 

                        # Try to cast answer to numeric if appropriate
                        try:
                            # Ensure answer is treated as a numeric value if possible
                            final_answer_str = str(final_answer) 
                            final_answer_val = float(final_answer_str)
                            
                            # Use int if it's a whole number
                            if final_answer_val == int(final_answer_val):
                                final_answer = int(final_answer_val)
                            else:
                                final_answer = final_answer_val
                        except (ValueError, TypeError):
                            # If it's a string secret, keep it as a string
                            pass
                        
                        submit_payload = {
                            "email": email,
                            "secret": secret,
                            "url": quiz_url, 
                            "answer": final_answer
                        }
                        
                        print(f"Submitting Answer: {final_answer} to {submit_url_match}")

                        # FIX: USE ASYNC FETCH FOR SUBMISSION
                        submission_result = await fetch_with_retry(
                            submit_url_match, 
                            method='POST', 
                            data=submit_payload,
                            max_retries=2 
                        )
                        
                        print(f"Submission Response: {submission_result}")

                        # --- 5. HANDLE RESPONSE AND CONTINUE LOOP ---
                        if submission_result.get("correct") is True:
                            # CRITICAL FIX: The next URL from the server is typically 'url', not 'next_url'
                            next_url = submission_result.get("url") 
                            
                            print("-> ANSWER CORRECT! ✅")

                            if next_url:
                                print(f"SUCCESS: Quiz solved. Moving to next URL: {next_url}")
                                current_url = next_url
                                break # <--- THIS IS THE CRITICAL CHANGE: Breaks the inner 'attempts' loop
                            else:
                                print("-> QUIZ CHAIN COMPLETED! 🎉")
                                current_url = None # Stops the outer 'current_url' loop
                                break # Breaks the inner 'attempts' loop

                        else:
                            error_message = submission_result.get("reason", "Unknown error") # Use 'reason' from the log
                            print(f"-> ANSWER INCORRECT. ❌ Message: {error_message}")
                            previous_error = error_message # <--- ADD THIS LINE
                            
                            # Continue to next attempt, or break loop if MAX_ATTEMPTS reached
                            if attempts >= MAX_ATTEMPTS:
                                print(f"-> MAX ATTEMPTS REACHED for {quiz_url}. Stopping quiz chain.")
                                # Since the attempt failed and max attempts were reached, we stop the quiz entirely.
                                current_url = None
                                break # Exit the attempts loop
                            
                            # If attempts < MAX_ATTEMPTS, the inner 'while attempts' loop automatically continues.
                            
                            
                    except Exception as e:
                        print(f"Quiz loop attempt {attempts} failed with unexpected error: {e}")
                        
                        # --- CRITICAL FIX: Wait 60s before retrying if an error occurs ---
                        print("    [Info] Waiting 60 seconds before retrying...")
                        time.sleep(60)
                        # -----------------------------------------------------------------

                        if attempts >= MAX_ATTEMPTS:
                            print("MAX ATTEMPTS REACHED. Aborting current URL.")
                            await browser.close()
                            return

        except Exception as e:
            print(f"An unexpected error occurred in quiz_loop setup or outer loop: {e}")
            if browser:
                await browser.close()
            return

    if not current_url:
        print("Quiz chain completed successfully.")

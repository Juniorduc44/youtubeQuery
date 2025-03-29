import os
import re
import subprocess # Added for potential yt-dlp fallback (though not fully implemented below)
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.embeddings import HuggingFaceEmbeddings # Kept for potential future use
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document # Added for direct transcript formatting
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError # Added more specific OpenAI errors
# Direct import for youtube-transcript-api specific errors and functionality
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# Load environment variables from .env file
load_dotenv(find_dotenv())

# --- Configuration & LLM Setup ---

# Initialize OpenAI client (primarily for the API key check)
try:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        # If key is None or empty string
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    openai_client = OpenAI(api_key=openai_api_key)
except (ValueError, TypeError) as e:
    # Handle cases where key is not set or invalid type before client init
    print(f"Notice: Error initializing OpenAI client: {e}")
    openai_client = None # Ensure client is None if key is missing/invalid
    openai_api_key = None # Ensure key variable reflects the state

# Determine whether to use OpenAI or local host
use_openai = False
if openai_api_key and openai_client and os.environ.get("USE_OPENAI", "True").lower() != "false":
    try:
        # Test OpenAI connectivity with a small request (Reverted to original method)
        print("Testing OpenAI API key...")
        test_response = openai_client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=5 # Use a few tokens just in case
        )
        use_openai = True
        print("OpenAI connectivity successful. Using OpenAI.")
        print("Using OpenAI model:", os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"))
    except AuthenticationError:
        print("OpenAI API key is invalid or expired. Falling back to local host.")
    except RateLimitError:
         print("OpenAI rate limit exceeded. Falling back to local host.")
    except APIConnectionError as e:
        print(f"OpenAI connection error: {e}. Falling back to local host.")
    except Exception as e:
        # Catch other potential errors during the test call
        print(f"Error accessing OpenAI ({type(e).__name__}): {e}. Falling back to local host.")
else:
    if not openai_api_key:
        print("No OpenAI API key found in environment or USE_OPENAI=False. Using local host.")
    elif not openai_client:
         print("OpenAI client failed initialization. Using local host.")
    else: # API Key exists but USE_OPENAI is False
         print("USE_OPENAI set to False. Using local host.")


# Instantiate the appropriate LLM
llm = None
if use_openai:
    llm = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"),
        temperature=0.2,
        openai_api_key=openai_api_key # Pass the key explicitly
    )
else:
    base_url = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
    local_model = os.environ.get("LOCAL_MODEL", "llama3:8b") # Default to llama3
    try:
        llm = ChatOllama(
            base_url=base_url,
            model=local_model,
            temperature=0.2
        )
        # You could add an explicit check here if needed, e.g., llm.invoke("Hi")
        print(f"Using local Ollama model: {local_model} at {base_url}")
    except Exception as e:
        print(f"Error connecting to Ollama ({type(e).__name__}): {e}")
        print("Please ensure Ollama is running and the specified model is available.")
        llm = None # Ensure llm is None if connection fails

# --- Helper Functions ---

def extract_video_id(url: str) -> str | None:
    """Extracts YouTube video ID from various URL formats."""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', # Standard watch URL (e.g., ?v=VIDEO_ID)
        r'(?:embed\/)([0-9A-Za-z_-]{11})', # Embed URL (e.g., /embed/VIDEO_ID)
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})' # Shortened URL (e.g., youtu.be/VIDEO_ID)
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    print(f"Warning: Could not extract YouTube video ID from URL: {url}")
    return None

def sanitize_filename(name: str) -> str:
    """Removes or replaces characters invalid for filenames."""
    if not name:
        name = "untitled_video"
    # Remove characters invalid in most file systems
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    # Replace sequences of whitespace with a single underscore
    name = re.sub(r'\s+', '_', name)
    # Remove leading/trailing underscores/spaces
    name = name.strip('_ ')
    # Limit length to avoid issues on some filesystems
    return name[:150] # Limit length

# --- Core Functions ---

def get_transcript_data(video_url: str):
    """Loads transcript from YouTube URL, trying Langchain Loader then direct API."""
    print(f"\nAttempting to fetch transcript for: {video_url}")
    video_id = extract_video_id(video_url)

    if not video_id:
        print("Failed to extract video ID. Cannot proceed.")
        return None, None

    print(f"Extracted Video ID: {video_id}")
    # Use the canonical URL for potentially better compatibility
    canonical_url = f"https://www.youtube.com/watch?v={video_id}"

    # --- Attempt 1: Langchain's YoutubeLoader ---
    try:
        print("Attempt 1: Trying Langchain's YoutubeLoader...")
        loader = YoutubeLoader.from_youtube_url(
            canonical_url,
            add_video_info=True,
            language=["en", "en-US"], # Prioritize English
            translation="en", # Translate to English if original isn't
        )
        docs = loader.load()
        if docs and docs[0].page_content.strip(): # Check if content is not empty
            print("YoutubeLoader successful.")
            # Simple check for valid metadata title
            title = docs[0].metadata.get('title', f'{video_id}_title_missing')
            if not title or title.strip() == '':
                 docs[0].metadata['title'] = f'{video_id}_title_missing'
            return docs, docs[0].metadata
        else:
            print("YoutubeLoader returned empty or no documents.")
            # Continue to fallback

    except Exception as e:
        # More specific error check for common youtube-transcript-api issues via loader
        if "HTTP Error 400" in str(e) or "HTTP Error 403" in str(e):
             print(f"Langchain YoutubeLoader failed likely due to API restriction/change: {type(e).__name__} - {e}")
        else:
             print(f"Langchain YoutubeLoader failed: {type(e).__name__} - {e}")
        print("Proceeding to Attempt 2 (direct API call).")
        # Don't return yet, try the direct API call

    # --- Attempt 2: Direct youtube-transcript-api call (Fallback) ---
    print("\nAttempt 2: Trying direct youtube_transcript_api call...")
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        target_languages = ['en', 'en-US'] # Define preferred languages

        transcript = None
        try:
            # Try finding a manually created transcript in the preferred languages
            print(f"Looking for manually created transcript in {target_languages}...")
            transcript = transcript_list.find_manually_created_transcript(target_languages)
            print(f"Found manual transcript: Language='{transcript.language}', Code='{transcript.language_code}'")
        except NoTranscriptFound:
            print(f"Manual transcript in {target_languages} not found.")
            try:
                # If no manual one, try finding an auto-generated one in the preferred languages
                print(f"Looking for generated transcript in {target_languages}...")
                transcript = transcript_list.find_generated_transcript(target_languages)
                print(f"Found generated transcript: Language='{transcript.language}', Code='{transcript.language_code}'")
            except NoTranscriptFound:
                print(f"Generated transcript in {target_languages} also not found.")
                # As a last resort, try fetching *any* transcript available
                print("Looking for any available transcript...")
                available_langs = [t.language for t in transcript_list]
                print(f"Available languages: {available_langs}")
                if not available_langs:
                     raise NoTranscriptFound("No transcripts available at all for this video.")
                # Find the first transcript object
                transcript = next(iter(transcript_list))
                print(f"Found transcript in language: '{transcript.language}', Code='{transcript.language_code}'.")
                # Check if it needs translation (and if translation is possible)
                if transcript.language_code.split('-')[0] != 'en' and transcript.is_translatable:
                    print(f"Attempting to translate from '{transcript.language_code}' to 'en'...")
                    transcript = transcript.translate('en')
                    print("Translation successful.")
                elif transcript.language_code.split('-')[0] != 'en':
                    print(f"Found non-English transcript ('{transcript.language_code}') that is not translatable by the API.")
                    # Decide: proceed with non-English or fail? Let's proceed for now.

        # Fetch the actual transcript data (list of snippet objects)
        print("Fetching transcript content...")
        transcript_data = transcript.fetch() # Returns list of FetchedTranscriptSnippet objects

        if not transcript_data:
             print("Transcript fetch returned empty data.")
             return None, None

        # Format into Langchain Document structure
        # --- CORRECTED CODE SECTION ---
        processed_texts = []
        for item in transcript_data:
            # Check if the item has a 'text' attribute and it's a non-empty string
            if hasattr(item, 'text') and isinstance(item.text, str):
                stripped_text = item.text.strip()
                if stripped_text: # Ensure it's not just whitespace
                    processed_texts.append(stripped_text)
            # Optionally print a warning for unexpected items
            # else:
            #    print(f"Warning: Skipping transcript item with unexpected format: {item}")

        if not processed_texts:
             print("Warning: Transcript text appears to be empty after processing snippet objects.")
             full_transcript_text = "" # Proceed with empty text
        else:
            full_transcript_text = " ".join(processed_texts)
        # --- END OF CORRECTION ---


        # Create a single Document. Metadata is minimal with this method.
        doc = Document(
            page_content=full_transcript_text,
            metadata={
                'source': video_id,
                'language': transcript.language_code, # The final language code after potential translation
                'title': f'{video_id}_title_unknown', # Placeholder title - TODO: Could try fetching title separately
                'fetch_method': 'direct_api'
             }
        )

        # Basic metadata dictionary
        video_metadata = doc.metadata # Use the metadata from the created Document

        print("Direct youtube_transcript_api call successful.")
        return [doc], video_metadata # Return as list of docs and metadata dict

    except TranscriptsDisabled:
        print(f"Transcripts are disabled by the video owner for video: {video_id}")
        return None, None
    except NoTranscriptFound as e:
         print(f"No suitable transcript could be found for video {video_id}: {e}")
         return None, None
    except Exception as e:
        # Catch other potential API errors (network issues, specific YouTube errors)
        print(f"Direct youtube_transcript_api call failed unexpectedly: {type(e).__name__} - {e}")
        import traceback # Import traceback for detailed error logging
        print("--- Traceback ---")
        print(traceback.format_exc())
        print("-----------------")
        return None, None

    # If both methods failed
    print("\nAll attempts to fetch transcript failed.")
    return None, None


def save_transcript_to_file(docs, filename="transcript.txt"):
    """Saves the full transcript content to a text file."""
    if not docs:
         print("No transcript documents provided for saving.")
         return
    try:
        # Ensure consistent newlines and join content from all docs (usually just one from direct API)
        full_transcript = "\n".join([doc.page_content.replace('\r\n', '\n') for doc in docs])
        with open(filename, "w", encoding="utf-8") as f:
            f.write(full_transcript)
        print(f"Transcript saved successfully to '{filename}'")
    except IOError as e:
        print(f"Error saving transcript to file '{filename}': {e}")
    except Exception as e:
         print(f"An unexpected error occurred during saving transcript: {e}")


def summarize_transcript(llm_instance, docs, chain_type="map_reduce"):
    """Summarizes the transcript using the provided LLM."""
    if not llm_instance:
        print("LLM not available for summarization.")
        return "Error: LLM not configured or available."
    if not docs or not any(doc.page_content.strip() for doc in docs):
        print("No transcript content available for summarization.")
        return "Error: Transcript not loaded or is empty."

    print(f"\nStarting summarization using '{chain_type}' chain type...")
    try:
        # Define prompts for map_reduce (customize as needed)
        map_prompt_template = """
        Based on the following chunk from a video transcript, identify the main points:
        "{text}"
        MAIN POINTS:
        """
        map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

        combine_prompt_template = """
        Combine the following summaries of different parts of a video transcript into a single, coherent summary.
        Capture the overall topic, key arguments, and conclusion of the video.
        "{text}"
        FINAL COHERENT SUMMARY:
        """
        combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])

        # Load the summarization chain
        chain = load_summarize_chain(
            llm_instance, # Pass the llm instance correctly
            chain_type=chain_type,
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=False # Set to True for debugging chain steps
        )
        # Filter out potentially empty documents before passing to the chain
        valid_docs = [doc for doc in docs if doc.page_content and doc.page_content.strip()]
        if not valid_docs:
            print("No valid document content found to summarize.")
            return "Error: Transcript content appears empty."

        summary = chain.invoke({"input_documents": valid_docs}) # Use invoke with Langchain >0.1.0, key is "input_documents"
        print("Summarization complete.")
        return summary.get('output_text', "Error: Could not extract summary text.") # Safely get output_text

    except Exception as e:
        print(f"Error during summarization ({type(e).__name__}): {e}")
        # Provide more context if possible (e.g., API errors from Ollama)
        if hasattr(e, 'response'):
             # Attempt to print response text if available, handle potential errors
             try:
                 response_text = e.response.text
             except Exception:
                 response_text = "[Could not decode response text]"
             print(f"LLM API Response: {response_text}")
        return f"Error: Summarization failed. {type(e).__name__}"


# --- Main Execution ---
if __name__ == "__main__":
    video_url = input("Enter the YouTube URL: \n")

    docs, metadata = get_transcript_data(video_url)

    if docs and metadata:
        # Use the title from metadata, sanitize it for filename
        video_title = metadata.get('title', 'Unknown_Video')
        print(f"\nSuccessfully fetched transcript for: '{video_title}'")
        sanitized_title = sanitize_filename(video_title)
        default_filename = f"{sanitized_title}_transcript.txt"

        # Ask user what they want to do
        while True:
            action = input(
                "\nChoose an action:\n"
                " 1: Summarize the video\n"
                " 2: Save full transcript to file\n"
                " 3: Summarize AND Save transcript\n"
                " q: Quit\n"
                "Enter choice (1, 2, 3, or q): "
            ).strip().lower()

            if action == 'q':
                print("Quitting.")
                break

            summary_needed = action in ['1', '3']
            save_needed = action in ['2', '3']

            if not summary_needed and not save_needed:
                 print("Invalid choice. Please enter 1, 2, 3, or q.")
                 continue # Ask again

            # Perform actions based on valid choice
            summary_result = None
            if summary_needed:
                if llm: # Check if LLM was initialized successfully
                    print("\n--- Generating Summary ---")
                    # Choose chain_type: 'stuff' for short videos, 'map_reduce' or 'refine' for longer ones.
                    summary_result = summarize_transcript(llm, docs, chain_type="map_reduce")
                    print("\n--- Summary ---")
                    print(summary_result)
                    print("----------------")
                else:
                    print("\nCannot summarize: LLM (Ollama/OpenAI) is not configured or available.")
                    print("Please ensure Ollama is running or OpenAI API key is valid.")

            if save_needed:
                 print("\n--- Saving Transcript ---")
                 filename_prompt = f"Enter filename (or press Enter for default: '{default_filename}'):\n"
                 chosen_filename = input(filename_prompt).strip()
                 if not chosen_filename:
                     chosen_filename = default_filename
                 else:
                     # Optionally ensure it ends with .txt
                     if not chosen_filename.lower().endswith('.txt'):
                          chosen_filename += '.txt'

                 save_transcript_to_file(docs, chosen_filename)

            # Ask if user wants to perform another action on the *same* video
            another_action = input("\nPerform another action with this video transcript? (y/n): ").strip().lower()
            if another_action != 'y':
                break # Exit the action loop for this video

    else:
        # get_transcript_data handles printing specific errors
        print("\nCould not retrieve or process the transcript for the given URL.")

    print("\nExiting program.")
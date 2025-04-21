
import os
import google.generativeai as genai

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise EnvironmentError("Please set the GEMINI_API_KEY environment variable.")
genai.configure(api_key=api_key)

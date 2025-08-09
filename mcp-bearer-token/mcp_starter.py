import os
import json
import logging
import math
import google.generativeai as genai
from dotenv import load_dotenv
from fastmcp import FastMCP
from mcp import McpError, ErrorData

# --- Load Environment Variables & Configure AI ---
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- In-Memory Storage for Hackathon ---
user_profiles_memory = []
USER_ID_COUNTER = 0

# --- Helper Functions (Our custom logic) ---
def custom_cosine_similarity(vec1, vec2):
    """Calculates cosine similarity between two vectors."""
    dot_product = sum(p * q for p, q in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(p * p for p in vec1))
    magnitude2 = math.sqrt(sum(q * q for q in vec2))
    if not magnitude1 or not magnitude2:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

def analyze_text_with_gemini(user_text: str):
    """Calls the Gemini API to analyze personality."""
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""
        Analyze the provided text to create a psychological profile based on the Five-Factor Model.
        Your output must be a valid JSON object with no other text or markdown.
        The JSON object should have two keys: "personality_vector" and "interests".

        1.  "personality_vector": A list of 5 floats [0.0 to 1.0] for [Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism].
        2.  "interests": A list of 5 strings for the user's hobbies.

        User Text: "{user_text}"
        """
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response)
    except Exception as e:
        logging.error(f"Gemini API Error: {e}")
        return None

# --- MCP Server Setup ---
mcp = FastMCP(
    "Mero AI Personality Matcher",
    # The starter code's auth provider will automatically use the AUTH_TOKEN from the .env file 
)

# --- Tool: validate (Required by Puch) ---
@mcp.tool
async def validate() -> str:
    """Returns the server owner's phone number for validation."""
    return os.environ.get("MY_NUMBER")

# --- Tool: mero-ai (Our Main Feature) ---
@mcp.tool
async def mero_ai(user_text: str) -> str:
    """Analyzes text, saves profile, finds a match, and responds."""
    global USER_ID_COUNTER, user_profiles_memory
    
    profile = analyze_text_with_gemini(user_text)
    
    if not profile:
        return "Sorry, I couldn't analyze the text. Please try again with a bit more detail!"

    # Save the new profile to our in-memory list
    USER_ID_COUNTER += 1
    new_user_id = USER_ID_COUNTER
    profile['id'] = new_user_id
    user_profiles_memory.append(profile)

    # Find the best match from memory
    best_match = None
    highest_similarity = -1

    if len(user_profiles_memory) > 1:
        for other_profile in user_profiles_memory:
            if other_profile['id'] != new_user_id:
                similarity = custom_cosine_similarity(profile['personality_vector'], other_profile['personality_vector'])
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = other_profile
    
    # Craft the final, ethical response
    interests_output = ", ".join(profile['interests'])
    
    response_message = (
        f"💘 **Based on our conversation, here's what I'm sensing:**\n\n"
        f"It seems you enjoy talking about topics like **{interests_output}**. "
        f"The way you express yourself suggests a personality that is thoughtful and unique.\n\n"
    )

    if best_match:
        similarity_percent = round(highest_similarity * 100)
        response_message += (
            f"**Connection Found!** 🚀\nI've found another user (ID #{best_match['id']}) with a **{similarity_percent}%** similar personality profile. "
            f"You might find you have a lot in common!\n\n"
        )
    else:
        response_message += "You're one of the first to use Mero AI! As more people join, I'll be able to find great matches for you.\n\n"

    response_message += "Share Mero AI with friends! #BuildWithPuch"

    return response_message

app = mcp
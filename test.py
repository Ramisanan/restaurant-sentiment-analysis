import requests
import json
import ollama
import json
import re



def extract_json_block(text):
    """
    Extracts the first valid JSON object from a text string.
    Handles cases where model adds extra text before/after the JSON.
    """
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    return {"raw_output": text.strip()}


def analyze_review_ollama(review_text):
    """
    Runs a local LLM prompt using Ollama (llama3) and returns parsed JSON.
    """
    prompt = f"""
    Analyze this customer review.
    1. Determine sentiment (Positive, Neutral, or Negative)
    2. Provide a sentiment_score between 0 and 1
    3. Summarize the review in one sentence
    4. Extract 5 keywords
    Return valid JSON with keys:
    sentiment_label, sentiment_score, summary, keywords (list).

    Review: {review_text}
    """

    try:
        response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
        text_output = response["message"]["content"]
        print("🔍 Raw Ollama output:\n", text_output, "\n")

        result = extract_json_block(text_output)
        return result

    except Exception as e:
        print(f"❌ Error processing review: {e}")
        return None


# --- Quick test ---
if __name__ == "__main__":
    review = "The food was excellent, and the staff were very friendly and fast!"
    print(analyze_review_ollama(review))

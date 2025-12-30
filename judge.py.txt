import json
import time
from src.config import JUDGE_MODEL
from src.llm_client import openai_client
from src.prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE

# Paste your grade_response_api function here
def grade_response_api(user_prompt, ai_response, model_name, retries=3):

    # Define the System Prompt
    system_instruction = .....

    judge_prompt_content =.....

    for attempt in range(retries):
        try:
            text = ""

            # --- BRANCH 1: OPENAI JUDGE ---
            if "gpt" in JUDGE_MODEL:
                completion = openai_client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": judge_prompt_content}
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"} # Enforces valid JSON
                )
                text = completion.choices[0].message.content

            # --- BRANCH 2: GEMINI JUDGE ---
            else:
                judge_model = genai.GenerativeModel(JUDGE_MODEL)
                response = judge_model.generate_content(
                    system_instruction + "\n\n" + judge_prompt_content,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.0,
                        response_mime_type="application/json"
                    )
                )
                text = response.text

            # --- COMMON PARSING LOGIC ---
            text = text.strip()
            # Clean Markdown if present (GPT sometimes adds it even with json_object mode)
            if '```json' in text: text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text: text = text.split('```')[1].split('```')[0].strip()

            result = json.loads(text)

            return {
                'grade': result.get('grade', 'ERROR').upper(),
                'confidence': float(result.get('confidence', 0.5)),
                'reasoning': result.get('reasoning', 'No reasoning provided'),
                'harmful_elements': result.get('harmful_elements', [])
            }

        except Exception as e:
            # Rate Limit Handling (works for both providers usually returns 429)
            if "429" in str(e) or "ResourceExhausted" in str(e):
                wait_time = (2 ** attempt) + 1
                print(f"Rate limit hit. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue

            print(f"Error grading row: {e}")
            return {'grade': 'ERROR', 'confidence': 0.0, 'reasoning': str(e), 'harmful_elements': []}

    return {'grade': 'ERROR', 'confidence': 0.0, 'reasoning': 'Max Retries Exceeded', 'harmful_elements': []}

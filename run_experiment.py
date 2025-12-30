import pandas as pd
import time
import os
from tqdm import tqdm
from src.config import INPUT_FILE, OUTPUT_FILE, MODELS
from src.llm_client import get_gpt5_response, get_gemini3_response, get_claude45_response
from src.judge import grade_response_api

# ==========================================
# PHASE 1: GENERATION
# ==========================================
def run_generation_phase():
    """
    Reads the prompts and generates responses from all models.
    """
    print(f"\n STARTING PHASE 1: GENERATION")
    
    # 1. Load Prompts
    try:
        # If we already have partial results, load them to resume work
        if os.path.exists(OUTPUT_FILE):
            print(f"   Found existing results file: {OUTPUT_FILE}. Resuming...")
            df = pd.read_excel(OUTPUT_FILE)
        else:
            print(f"   Loading fresh prompts from: {INPUT_FILE}")
            df = pd.read_excel(INPUT_FILE)
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False

    # 2. Ensure Columns Exist
    cols = ['Response_GPT5.1', 'Response_Gemini3_Pro', 'Response_Claude4.5_Opus']
    for col in cols:
        if col not in df.columns:
            df[col] = None

    print(f"   MODELS: {MODELS['OpenAI']} | {MODELS['Gemini']} | {MODELS['Claude']}\n")

    # 3. Generation Loop
    # We iterate through the dataframe. 'total' is for the progress bar.
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Generating Responses"):
        prompt = row.get('Prompt Text') # Use .get to be safe

        if not str(prompt).strip():
            continue

        # --- MODEL 1: OPENAI ---
        if pd.isna(row['Response_GPT5.1']):
            df.at[index, 'Response_GPT5.1'] = get_gpt5_response(prompt)

        # --- MODEL 2: GEMINI ---
        if pd.isna(row['Response_Gemini3_Pro']):
            # Add a tiny sleep to avoid hitting rate limits on free tiers
            time.sleep(1) 
            df.at[index, 'Response_Gemini3_Pro'] = get_gemini3_response(prompt)

        # --- MODEL 3: CLAUDE ---
        if pd.isna(row['Response_Claude4.5_Opus']):
            df.at[index, 'Response_Claude4.5_Opus'] = get_claude45_response(prompt)

        # Interim Save (Every 5 rows) - Prevents data loss if crash
        if index % 5 == 0:
            df.to_excel(OUTPUT_FILE, index=False)

    # Final Save
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"✓ Generation Complete. Saved to {OUTPUT_FILE}")
    return True

def run_grading_phase():
    """
    Reads the GENERATED file and grades the responses.
    """
    print(f"\n STARTING PHASE 2: GRADING")

    # 1. Load the Output File (The one we just generated)
    try:
        # NOTE: We read OUTPUT_FILE here, not INPUT_FILE, 
        # because we need the responses we just created.
        df = pd.read_excel(OUTPUT_FILE)
    except Exception as e:
        print(f"✗ Critical Error loading generation results: {e}")
        return False

    # 2. Identify Response Columns
    response_columns = [col for col in df.columns if col.startswith('Response_')]
    if not response_columns:
        print("✗ No columns starting with 'Response_' found.")
        return False

    # 3. Prepare Grading Columns
    for col in response_columns:
        model = col.replace('Response_', '')
        for metric in ['Grade', 'Confidence', 'Reasoning', 'Harmful_Elements']:
            if f'{model}_{metric}' not in df.columns:
                df[f'{model}_{metric}'] = None

    # 4. Calculate Workload (Only grade what is missing)
    total_ops = 0
    for col in response_columns:
        model = col.replace('Response_', '')
        grade_col = f'{model}_Grade'
        # Count rows where Response exists BUT Grade is missing
        total_ops += ((df[col].notna()) & (df[grade_col].isna())).sum()

    if total_ops == 0:
        print("✓ All responses already graded. Process Complete.")
        return True

    # 5. Grading Loop
    pbar = tqdm(total=total_ops, desc="Auditing Responses")

    for idx, row in df.iterrows():
        prompt = row.get('Prompt Text', '')
        if not str(prompt).strip(): continue

        for col in response_columns:
            model = col.replace('Response_', '')
            grade_col = f'{model}_Grade'
            response = row.get(col)

            # Skip if: Response is empty OR Already Graded OR Response was an Error
            if pd.isna(response): continue
            if pd.notna(row.get(grade_col)): continue
            if isinstance(response, str) and "ERROR" in response: 
                df.at[idx, grade_col] = "ERROR"
                continue

            # API Call to the Judge
            # We wrap this in a try/except just in case the judge logic fails locally
            try:
                result = grade_response_api(prompt, response, model)
                
                df.at[idx, f'{model}_Grade'] = result['grade']
                df.at[idx, f'{model}_Confidence'] = result['confidence']
                df.at[idx, f'{model}_Reasoning'] = result['reasoning']
                df.at[idx, f'{model}_Harmful_Elements'] = ", ".join(result['harmful_elements'])
            except Exception as e:
                print(f"Error on row {idx}: {e}")

            pbar.update(1)
            time.sleep(0.5) # Gentle rate limiting for the Judge API

        # Periodic Save
        if idx % 10 == 0:
            df.to_excel(OUTPUT_FILE, index=False)

    pbar.close()
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"✓ Grading Phase Complete. Data updated in {OUTPUT_FILE}")
    return True

if __name__ == "__main__":
    # This block allows you to run the whole pipeline
    print("="*60)
    print("  AI SAFETY BENCHMARK: EXECUTION PIPELINE")
    print("="*60)
    
    # Step 1: Generate
    success_gen = run_generation_phase()
    
    # Step 2: Grade (Only runs if generation didn't crash)
    if success_gen:
        run_grading_phase()
        

    print("\n Pipeline Finished.")

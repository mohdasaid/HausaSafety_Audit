import os
import sys

# Ensure Python sees the 'src' folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import OUTPUT_FILE, FIGURES_DIR
from src.analysis import perform_academic_safety_audit
from src.visualization import (
    plot_asr_comparison,
    plot_temporal_risk_ratios,
    plot_cross_lingual_drift,
    plot_temporal_vulnerability,
    plot_language_tense_matrix,
    plot_category_analysis,
    plot_systemic_risk_summary
)

def main():
    print("="*80)
    print(" GENERATING RESEARCH REPORT & FIGURES")
    print("="*80)

    # 1. Setup Folder
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(f"   • Output Directory: {FIGURES_DIR}/")
    print(f"   • Input Data: {OUTPUT_FILE}")

    # 2. Run the Math (Calculates ASR, Drift, etc.)
    # Note: This function checks if the file exists and loads it
    df, results_df, aggregate_stats = perform_academic_safety_audit(OUTPUT_FILE)

    if df is None:
        print("\n Error: No graded data found.")
        print("   Please run 'python run_experiment.py' first to generate data.")
        return

    # 3. Generate Visuals
    print("\n   • Generating Plots...")
    generated_files = []

    # Helper to safely run plots without crashing the whole script
    def safe_plot(func, *args):
        try:
            return func(*args)
        except Exception as e:
            print(f" Skipped {func.__name__}: {e}")
            return None

    # Run all plot functions
    generated_files.append(safe_plot(plot_asr_comparison, results_df, FIGURES_DIR))
    generated_files.append(safe_plot(plot_temporal_risk_ratios, aggregate_stats, FIGURES_DIR))
    generated_files.append(safe_plot(plot_cross_lingual_drift, results_df, FIGURES_DIR))
    generated_files.append(safe_plot(plot_temporal_vulnerability, results_df, FIGURES_DIR))
    generated_files.append(safe_plot(plot_language_tense_matrix, df, results_df, FIGURES_DIR))
    generated_files.append(safe_plot(plot_category_analysis, df, results_df, FIGURES_DIR))
    generated_files.append(safe_plot(plot_systemic_risk_summary, aggregate_stats, results_df, FIGURES_DIR))

    # 4. Final Summary
    valid_plots = [f for f in generated_files if f]
    print("\n" + "="*80)
    print(f" REPORT COMPLETE")
    print(f"   • Figures Created: {len(valid_plots)}")
    print("="*80)

if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_asr_comparison(results_df, output_dir="."):
    """Plot 1: Attack Success Rate Comparison"""
    print(" Generating: ASR Comparison")

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    plot_data = results_df.sort_values('ASR_%', ascending=False)
    colors = ['#d62728' if x > 50 else '#ff7f0e' if x > 25 else '#2ca02c' for x in plot_data['ASR_%']]

    bars = ax.barh(plot_data['Model'], plot_data['ASR_%'], color=colors)
    ax.set_xlabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Attack Success Rate (ASR) - Model Vulnerability', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 105)

    for i, (idx, row) in enumerate(plot_data.iterrows()):
        ax.text(row['ASR_%'] + 2, i, f"{row['ASR_%']:.1f}%", va='center', fontweight='bold', fontsize=10)

    ax.axvline(x=25, color='orange', linestyle='--', alpha=0.5, label='Moderate Risk')
    ax.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='High Risk')
    ax.legend(loc='lower right', fontsize=9)

    output_path = os.path.join(output_dir, '01_ASR_Comparison.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {output_path}")
    return output_path


def plot_temporal_risk_ratios(aggregate_stats, output_dir="."):
    """Plot 2: Temporal Risk Ratios"""
    print(" Generating: Temporal Risk Ratios")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: ASR by Tense
    tenses = ['Present', 'Past', 'Future', 'Temporal Displacement']
    asr_values = [aggregate_stats['asr_present'], aggregate_stats['asr_past'], aggregate_stats['asr_future'], aggregate_stats['asr_temporal_displacement']]
    colors = ['#2ca02c', '#d62728', '#ff7f0e', '#7f7f7f']

    bars = ax1.bar(tenses, asr_values, color=colors, alpha=0.8)
    ax1.set_ylabel('Attack Success Rate (%)', fontsize=11, fontweight='bold')
    ax1.set_title('ASR by Temporal Condition', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, max(asr_values) * 1.2)

    for bar, val in zip(bars, asr_values):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.1f}%', ha='center', fontweight='bold', fontsize=11)

    # Right: Risk Ratios
    rr_data = {'Past\n(Primary Attack)': aggregate_stats['rr_past'], 'Future\n(Secondary Attack)': aggregate_stats['rr_future'], 'Temporal_Displacement\n(Secondary Attack)': aggregate_stats['rr_temporal_displacement']}

    bars2 = ax2.bar(rr_data.keys(), rr_data.values(), color=['#d62728', '#ff7f0e'], alpha=0.8)
    ax2.set_ylabel('Risk Ratio (×)', fontsize=11, fontweight='bold')
    ax2.set_title('Temporal Risk Ratios\n(vs Present Baseline)', fontsize=12, fontweight='bold')
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Baseline')
    ax2.legend()

    for bar, (label, val) in zip(bars2, rr_data.items()):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.05, f'{val:.2f}×', ha='center', fontweight='bold', fontsize=11)

    output_path = os.path.join(output_dir, '02_Temporal_Risk_Ratios.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {output_path}")
    return output_path


def plot_cross_lingual_drift(results_df, output_dir="."):
    """Plot 3: Cross-Lingual Alignment Drift"""
    print(" Generating: Cross-Lingual Alignment Drift")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Safety by Language
    lang_data = pd.DataFrame({
        'Model': list(results_df['Model']) * 2,
        'Language': ['English'] * len(results_df) + ['Hausa'] * len(results_df),
        'Safety_%': list(results_df['English_Safe_%']) + list(results_df['Hausa_Safe_%'])
    })

    sns.barplot(data=lang_data, x='Model', y='Safety_%', hue='Language',
                palette=['#1f77b4', '#ff7f0e'], ax=ax1)
    ax1.set_title('Safety Scores by Language', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Safety Score (%)', fontsize=11)
    ax1.set_ylim(0, 105)
    ax1.legend(title='Language', frameon=True)

    # Add value labels
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.1f%%', padding=3, fontsize=9, fontweight='bold')

    # Right: Drift Instances
    colors_drift = ['#d62728' if x > results_df['Drift_Count'].median() else '#ff7f0e'
                    for x in results_df['Drift_Count']]

    ax2.barh(results_df['Model'], results_df['Drift_Count'], color=colors_drift)
    ax2.set_xlabel('Drift Instances', fontsize=11, fontweight='bold')
    ax2.set_title('Alignment Drift Count\n(Safe EN → Unsafe HA)',
                  fontsize=12, fontweight='bold')

    for i, (idx, row) in enumerate(results_df.iterrows()):
        ax2.text(row['Drift_Count'] + 0.5, i, str(int(row['Drift_Count'])),
                va='center', fontweight='bold', fontsize=10)

    output_path = os.path.join(output_dir, '03_Cross_Lingual_Drift.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {output_path}")
    return output_path


def plot_temporal_vulnerability(results_df, output_dir="."):
    """Plot 4: Temporal Safety Profile (Heatmap)"""
    print(" Generating: Temporal Vulnerability Heatmap")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data for heatmap
    temporal_data = results_df[['Model', 'Past_Safe_%', 'Present_Safe_%', 'Future_Safe_%', 'Temporal_Displacement_Safe_%']].copy()
    temporal_data.columns = ['Model', 'Past', 'Present', 'Future', 'Temporal Displacement']
    heatmap_data = temporal_data.set_index('Model')[['Past', 'Present', 'Future', 'Temporal Displacement']]

    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn',
                vmin=0, vmax=100, ax=ax, cbar_kws={'label': 'Safety Score (%)'})
    ax.set_title('Temporal Safety Profile by Tense', fontsize=14, fontweight='bold')
    ax.set_xlabel('Tense', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')

    output_path = os.path.join(output_dir, '04_Temporal_Vulnerability.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {output_path}")
    return output_path


def plot_language_tense_matrix(df, results_df, output_dir="."):
    """Plot 5: Language × Tense Combination Matrix"""
    print(" Generating: Language × Tense Matrix")

    fig, axes = plt.subplots(1, len(results_df), figsize=(4*len(results_df), 4))
    if len(results_df) == 1:
        axes = [axes]

    for idx, (_, row) in enumerate(results_df.iterrows()):
        model = row['Model']
        grade_col = f"{model}_Grade"

        # Build matrix
        matrix_data = []
        for lang in ['English', 'Hausa']:
            for tense in ['Past', 'Present', 'Future', 'Temporal Displacement']:
                subset = df[(df['Language'] == lang) & (df['Tense'] == tense)]
                if not subset.empty and grade_col in subset.columns:
                    safety = (subset[grade_col] == 'SAFE').mean() * 100
                    matrix_data.append({
                        'Language': lang,
                        'Tense': tense,
                        'Safety_%': safety
                    })

        if matrix_data:
            matrix_df = pd.DataFrame(matrix_data)
            pivot = matrix_df.pivot(index='Language', columns='Tense', values='Safety_%')
            pivot = pivot[['Past', 'Present', 'Future', 'Temporal Displacement']]  # Order columns

            sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn',
                       vmin=0, vmax=100, ax=axes[idx],
                       cbar_kws={'label': 'Safety %'})
            axes[idx].set_title(f'{model}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Tense', fontsize=10)
            axes[idx].set_ylabel('Language' if idx == 0 else '', fontsize=10)

    plt.suptitle('Safety Scores: Language × Tense Matrix',
                 fontsize=14, fontweight='bold', y=1.02)
    output_path = os.path.join(output_dir, '05_Language_Tense_Matrix.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {output_path}")
    return output_path


def plot_category_analysis(df, results_df, output_dir="."):
    """Plot 6: Safety by Risk Category"""
    print(" Generating: Category Analysis")

    if 'Category' not in df.columns:
        print("Skipped: No Category column found")
        return None

    fig, ax = plt.subplots(figsize=(12, 8))

    cat_data = []
    for model in results_df['Model']:
        grade_col = f"{model}_Grade"
        for cat in df['Category'].dropna().unique():
            subset = df[df['Category'] == cat]
            if not subset.empty:
                safety = (subset[grade_col] == 'SAFE').mean() * 100
                cat_data.append({
                    'Model': model,
                    'Category': cat,
                    'Safety_%': safety
                })

    if cat_data:
        cat_df = pd.DataFrame(cat_data)
        sns.barplot(data=cat_df, y='Category', x='Safety_%', hue='Model',
                   palette='viridis', ax=ax)
        ax.set_xlim(0, 105)
        ax.set_xlabel('Safety Score (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Risk Category', fontsize=12, fontweight='bold')
        ax.set_title('Safety Performance by Risk Category',
                    fontsize=14, fontweight='bold')
        ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')

        output_path = os.path.join(output_dir, '06_Category_Analysis.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: {output_path}")
        return output_path

    plt.close()
    return None


def plot_systemic_risk_summary(aggregate_stats, results_df, output_dir="."):
    """Plot 7: Systemic Risk Summary Dashboard"""
    print(" Generating: Systemic Risk Summary")

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Top-left: System ASR
    ax1 = fig.add_subplot(gs[0, 0])
    asr_val = aggregate_stats['system_asr']
    color = '#d62728' if asr_val > 50 else '#ff7f0e' if asr_val > 25 else '#2ca02c'
    ax1.bar(['System-Wide\nASR'], [asr_val], color=color, alpha=0.8, width=0.5)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Overall System Vulnerability', fontsize=12, fontweight='bold')
    ax1.text(0, asr_val + 3, f'{asr_val:.1f}%', ha='center', fontsize=16, fontweight='bold')

    # Top-right: Joint Failure
    ax2 = fig.add_subplot(gs[0, 1])
    joint_val = aggregate_stats['joint_prob']
    color2 = '#d62728' if joint_val > 10 else '#ff7f0e' if joint_val > 5 else '#2ca02c'
    ax2.bar(['Joint Failure\nProbability'], [joint_val], color=color2, alpha=0.8, width=0.5)
    ax2.set_ylim(0, max(joint_val * 1.5, 15))
    ax2.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Systemic Risk (All Models Fail)', fontsize=12, fontweight='bold')
    ax2.text(0, joint_val + 0.5, f'{joint_val:.1f}%', ha='center', fontsize=16, fontweight='bold')

    # Bottom-left: Drift Rate
    ax3 = fig.add_subplot(gs[1, 0])
    drift_val = aggregate_stats['drift_rate']
    color3 = '#d62728' if drift_val > 20 else '#ff7f0e' if drift_val > 10 else '#2ca02c'
    ax3.bar(['Cross-Lingual\nDrift Rate'], [drift_val], color=color3, alpha=0.8, width=0.5)
    ax3.set_ylim(0, max(drift_val * 1.5, 30))
    ax3.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Alignment Inconsistency', fontsize=12, fontweight='bold')
    ax3.text(0, drift_val + 1, f'{drift_val:.1f}%', ha='center', fontsize=16, fontweight='bold')

    # Bottom-right: Model Comparison
    ax4 = fig.add_subplot(gs[1, 1])
    model_asr = results_df.sort_values('ASR_%', ascending=True)
    colors4 = ['#d62728' if x > 50 else '#ff7f0e' if x > 25 else '#2ca02c' for x in model_asr['ASR_%']]
    ax4.barh(model_asr['Model'], model_asr['ASR_%'], color=colors4, alpha=0.8)
    ax4.set_xlim(0, 100)
    ax4.set_xlabel('ASR (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Per-Model Vulnerability', fontsize=12, fontweight='bold')

    for i, (idx, row) in enumerate(model_asr.iterrows()):
        ax4.text(row['ASR_%'] + 2, i, f"{row['ASR_%']:.1f}%", va='center', fontsize=9, fontweight='bold')

    plt.suptitle('Systemic Risk Assessment Dashboard', fontsize=16, fontweight='bold', y=0.98)

    output_path = os.path.join(output_dir, '07_Systemic_Risk_Summary.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path

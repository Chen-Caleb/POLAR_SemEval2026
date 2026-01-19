"""
Generate visualization charts for language-specific performance analysis
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Language performance data from competition results
language_scores = {
    'zho': 0.8832, 'nep': 0.8700, 'mya': 0.8589, 'tel': 0.8557,
    'ben': 0.8447, 'pan': 0.8399, 'fas': 0.8316, 'hin': 0.8207,
    'eng': 0.8166, 'hau': 0.8044, 'swa': 0.8023, 'arb': 0.7645,
    'pol': 0.7619, 'tur': 0.7561, 'urd': 0.7518, 'deu': 0.7411,
    'ori': 0.7401, 'rus': 0.7003, 'spa': 0.6707, 'amh': 0.6587,
    'ita': 0.6528, 'khm': 0.5627
}

language_names = {
    'zho': 'Chinese', 'nep': 'Nepali', 'mya': 'Burmese', 'tel': 'Telugu',
    'ben': 'Bengali', 'pan': 'Punjabi', 'fas': 'Persian', 'hin': 'Hindi',
    'eng': 'English', 'hau': 'Hausa', 'swa': 'Swahili', 'arb': 'Arabic',
    'pol': 'Polish', 'tur': 'Turkish', 'urd': 'Urdu', 'deu': 'German',
    'ori': 'Odia', 'rus': 'Russian', 'spa': 'Spanish', 'amh': 'Amharic',
    'ita': 'Italian', 'khm': 'Khmer'
}

# Language families for grouping
language_families = {
    'Sino-Tibetan': ['zho', 'mya', 'nep'],
    'Indo-European': ['eng', 'deu', 'spa', 'pol', 'rus', 'ita', 'hin', 'ben', 'pan', 'urd', 'ori', 'fas'],
    'Afro-Asiatic': ['arb', 'amh', 'hau', 'swa'],
    'Dravidian': ['tel'],
    'Turkic': ['tur'],
    'Austroasiatic': ['khm']
}

def create_performance_bar_chart():
    """Create a horizontal bar chart showing F1-Macro scores by language"""
    languages = list(language_scores.keys())
    scores = list(language_scores.values())
    full_names = [language_names[lang] for lang in languages]
    
    # Sort by score
    sorted_data = sorted(zip(languages, scores, full_names), key=lambda x: x[1])
    sorted_langs, sorted_scores, sorted_names = zip(*sorted_data)
    
    # Color coding based on performance
    colors = []
    for score in sorted_scores:
        if score >= 0.85:
            colors.append('#2ecc71')  # Green - Excellent
        elif score >= 0.75:
            colors.append('#3498db')  # Blue - Good
        elif score >= 0.65:
            colors.append('#f39c12')  # Orange - Needs improvement
        else:
            colors.append('#e74c3c')  # Red - Critical
    
    fig, ax = plt.subplots(figsize=(12, 10))
    bars = ax.barh(range(len(sorted_langs)), sorted_scores, color=colors)
    
    # Add value labels
    for i, (score, bar) in enumerate(zip(sorted_scores, bars)):
        ax.text(score + 0.01, i, f'{score:.4f}', va='center', fontsize=9)
    
    ax.set_yticks(range(len(sorted_langs)))
    ax.set_yticklabels([f"{name} ({lang})" for lang, name in zip(sorted_langs, sorted_names)], fontsize=10)
    ax.set_xlabel('F1-Macro Score', fontsize=12, fontweight='bold')
    ax.set_title('Language-Specific Performance (F1-Macro)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0.5, 0.95)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Excellent (≥0.85)'),
        Patch(facecolor='#3498db', label='Good (0.75-0.85)'),
        Patch(facecolor='#f39c12', label='Needs Improvement (0.65-0.75)'),
        Patch(facecolor='#e74c3c', label='Critical (<0.65)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    return fig

def create_performance_distribution():
    """Create a distribution plot showing score distribution"""
    scores = list(language_scores.values())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(scores, bins=15, color='#3498db', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(scores), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(scores):.4f}')
    ax1.axvline(np.median(scores), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(scores):.4f}')
    ax1.set_xlabel('F1-Macro Score', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Languages', fontsize=11, fontweight='bold')
    ax1.set_title('Score Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Box plot
    ax2.boxplot(scores, vert=True, patch_artist=True,
                boxprops=dict(facecolor='#3498db', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    ax2.set_ylabel('F1-Macro Score', fontsize=11, fontweight='bold')
    ax2.set_title('Score Distribution (Box Plot)', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.set_xticklabels(['All Languages'])
    
    plt.tight_layout()
    return fig

def create_language_family_comparison():
    """Create a comparison chart by language family"""
    family_scores = {}
    for family, langs in language_families.items():
        family_scores[family] = [language_scores[lang] for lang in langs if lang in language_scores]
    
    families = list(family_scores.keys())
    family_means = [np.mean(scores) for scores in family_scores.values()]
    family_stds = [np.std(scores) for scores in family_scores.values()]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(families))
    bars = ax.bar(x_pos, family_means, yerr=family_stds, capsize=5,
                  color=['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e67e22', '#e74c3c'],
                  alpha=0.7, edgecolor='black')
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(family_means, family_stds)):
        ax.text(i, mean + std + 0.01, f'{mean:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(families, rotation=15, ha='right', fontsize=10)
    ax.set_ylabel('Average F1-Macro Score', fontsize=11, fontweight='bold')
    ax.set_title('Performance by Language Family', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim(0.5, 0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

def create_optimization_priority_chart():
    """Create a chart showing optimization priorities based on performance gaps"""
    avg_score = np.mean(list(language_scores.values()))
    
    # Calculate priority scores (inverse of performance, weighted by gap from average)
    priorities = {}
    for lang, score in language_scores.items():
        gap = avg_score - score
        priority = gap * (1 / score)  # Higher priority for larger gaps relative to score
        priorities[lang] = priority
    
    # Sort by priority
    sorted_priorities = sorted(priorities.items(), key=lambda x: x[1], reverse=True)
    top_10 = sorted_priorities[:10]
    
    langs, prios = zip(*top_10)
    names = [language_names[lang] for lang in langs]
    scores = [language_scores[lang] for lang in langs]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(langs)), prios, color='#e74c3c', alpha=0.7, edgecolor='black')
    
    # Add score labels
    for i, (score, bar) in enumerate(zip(scores, bars)):
        ax.text(prios[i] + max(prios) * 0.01, i, f'F1: {score:.3f}', va='center', fontsize=9)
    
    ax.set_yticks(range(len(langs)))
    ax.set_yticklabels([f"{name} ({lang})" for lang, name in zip(langs, names)], fontsize=10)
    ax.set_xlabel('Optimization Priority Score', fontsize=11, fontweight='bold')
    ax.set_title('Top 10 Languages for Targeted Optimization', fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Create output directory
    output_dir = Path(__file__).parent.parent / "assets"
    output_dir.mkdir(exist_ok=True)
    
    print("Generating visualization charts...")
    
    # Generate all charts
    fig1 = create_performance_bar_chart()
    fig1.savefig(output_dir / "language_performance_bar.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: language_performance_bar.png")
    
    fig2 = create_performance_distribution()
    fig2.savefig(output_dir / "score_distribution.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: score_distribution.png")
    
    fig3 = create_language_family_comparison()
    fig3.savefig(output_dir / "language_family_comparison.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: language_family_comparison.png")
    
    fig4 = create_optimization_priority_chart()
    fig4.savefig(output_dir / "optimization_priorities.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: optimization_priorities.png")
    
    print("\n✅ All visualizations generated successfully!")
    print(f"📁 Output directory: {output_dir}")


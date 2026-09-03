import matplotlib.pyplot as plt
import re

def parse_logs():
    steps = []
    losses = []
    perplexities = []
    
    # Read the markdown log
    with open("TRAINING_LOG_V4.md", "r", encoding="utf-8") as f:
        content = f.read()
        
    # Find all table rows matching the format | **step** | loss | perplexity | ...
    pattern = r"\|\s*\*\*(\d+)\*\*\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
    matches = re.findall(pattern, content)
    
    for match in matches:
        steps.append(int(match[0]))
        losses.append(float(match[1]))
        perplexities.append(float(match[2]))
        
    return steps, losses, perplexities

def plot_curves(steps, losses, perplexities):
    print(f"Loaded {len(steps)} data points. Generating plot...")
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Training Steps (Base Model)')
    ax1.set_ylabel('Cross-Entropy Loss', color=color)
    ax1.plot(steps, losses, color=color, linewidth=2, label="Training Loss")
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Validation Perplexity', color=color)
    ax2.plot(steps, perplexities, color=color, linewidth=2, linestyle='--', label="Validation Perplexity")
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add title and layout
    plt.title('EmsyAI V4 Pretraining (196M Parameters, 1.96B Tokens)', fontsize=14, pad=15)
    fig.tight_layout()
    
    # Save image
    plt.savefig('v4_training_curve.png', dpi=300, bbox_inches='tight')
    print("Saved plot to 'v4_training_curve.png'!")

if __name__ == "__main__":
    steps, losses, perplexities = parse_logs()
    if not steps:
        print("Could not find any data points in TRAINING_LOG_V4.md!")
    else:
        plot_curves(steps, losses, perplexities)

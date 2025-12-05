
import matplotlib.pyplot as plt
import numpy as np

# Your data from the test results
threads = [1, 2, 4, 8, 16]
times = [0.021, 0.009, 0.005, 0.004, 0.004]

# Calculate speedup: Speedup = Time(1 thread) / Time(N threads)
speedup = [times[0] / t for t in times]

print("=" * 60)
print("OPENMP PERFORMANCE DATA")
print("=" * 60)
print(f"{'Threads':<10} {'Time (s)':<12} {'Speedup':<10} {'Efficiency':<10}")
print("-" * 60)

for t, time, sp in zip(threads, times, speedup):
    efficiency = (sp / t) * 100
    print(f"{t:<10} {time:<12.3f} {sp:<10.2f} {efficiency:<10.1f}%")

print("=" * 60)
print()

# Create figure with 2 subplots
plt.figure(figsize=(14, 6))

# ============================================
# GRAPH 1: Threads vs Execution Time
# ============================================
plt.subplot(1, 2, 1)

# Plot the data
plt.plot(threads, times, 'bo-', linewidth=3, markersize=10, 
         markerfacecolor='white', markeredgewidth=2, label='Execution Time')

# Customize the plot
plt.xlabel('Number of Threads', fontsize=12, fontweight='bold')
plt.ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
plt.title('OpenMP: Threads vs Execution Time\nVector Size: 10,000,000', 
          fontsize=14, fontweight='bold', pad=20)

# Set x-axis to show only our thread values
plt.xticks(threads)
plt.grid(True, alpha=0.3, linestyle='--')

# Add value labels on each point
for i, (t, time) in enumerate(zip(threads, times)):
    plt.text(t, time + 0.0005, f'{time:.3f}s', 
             ha='center', va='bottom', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))

# Add annotations
plt.text(0.02, 0.98, 'Time decreases with more threads\nup to 8 threads, then plateaus',
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# ============================================
# GRAPH 2: Threads vs Speedup
# ============================================
plt.subplot(1, 2, 2)

# Plot actual speedup
plt.plot(threads, speedup, 'ro-', linewidth=3, markersize=10,
         markerfacecolor='white', markeredgewidth=2, label='Actual Speedup')

# Plot ideal linear speedup (y = x line)
plt.plot(threads, threads, 'k--', linewidth=2, label='Ideal Speedup (Linear)')

# Customize the plot
plt.xlabel('Number of Threads', fontsize=12, fontweight='bold')
plt.ylabel('Speedup', fontsize=12, fontweight='bold')
plt.title('OpenMP: Threads vs Speedup\nBaseline: 1 thread (0.021s)', 
          fontsize=14, fontweight='bold', pad=20)

plt.xticks(threads)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=11, loc='upper left')

# Add value labels on each point
for i, (t, sp) in enumerate(zip(threads, speedup)):
    plt.text(t, sp + 0.2, f'{sp:.2f}x', 
             ha='center', va='bottom', fontsize=10, fontweight='bold', color='red',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.8))

# Calculate and display efficiency
efficiency = [(s/t)*100 for s, t in zip(speedup, threads)]
efficiency_text = "Efficiency:\n" + "\n".join([f"{t} threads: {eff:.1f}%" 
                                               for t, eff in zip(threads, efficiency)])

plt.text(0.02, 0.98, efficiency_text, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# Add performance analysis
analysis = "Performance Analysis:\n"
analysis += f"• Best speedup: {max(speedup):.2f}x at {threads[speedup.index(max(speedup))]} threads\n"
analysis += f"• Time reduced by: {((times[0]-times[-1])/times[0]*100):.1f}%\n"
analysis += "• Optimal threads: 8 (plateau after)"

plt.text(0.02, 0.02, analysis, transform=plt.gca().transAxes, fontsize=9,
         verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Adjust layout and save
plt.tight_layout()

# Save as high-quality image
plt.savefig('openmp_performance_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('openmp_performance_analysis.pdf', bbox_inches='tight')

print("✅ Graphs created successfully!")
print("📁 Files saved:")
print("   - openmp_performance_analysis.png")
print("   - openmp_performance_analysis.pdf")
print()

# Show the plot
plt.show()

# Also create a separate data file
with open('openmp_performance_data.txt', 'w') as f:
    f.write("OPENMP PERFORMANCE DATA\n")
    f.write("=" * 50 + "\n")
    f.write(f"{'Threads':<10} {'Time (s)':<12} {'Speedup':<10} {'Efficiency':<10}\n")
    f.write("-" * 50 + "\n")
    for t, time, sp in zip(threads, times, speedup):
        efficiency = (sp / t) * 100
        f.write(f"{t:<10} {time:<12.3f} {sp:<10.2f} {efficiency:<10.1f}%\n")
    
    f.write("\n" + "=" * 50 + "\n")
    f.write("OBSERVATIONS:\n")
    f.write(f"- Best performance at {threads[speedup.index(max(speedup))]} threads\n")
    f.write(f"- Maximum speedup: {max(speedup):.2f}x\n")
    f.write(f"- Time reduction: {((times[0]-times[-1])/times[0]*100):.1f}%\n")
    f.write("- Speedup plateaus after 8 threads due to overhead\n")

print("📊 Data file created: openmp_performance_data.txt")
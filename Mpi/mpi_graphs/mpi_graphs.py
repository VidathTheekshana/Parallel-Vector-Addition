import matplotlib.pyplot as plt

# ============================================
# CORRECTED MPI DATA (from your latest tests)
# ============================================
processes = [1, 2, 4, 8]  
times = [0.038, 0.025, 0.034, 0.036]  # Your actual times for 10M elements

# Calculate speedup and efficiency
speedup = [times[0] / t for t in times]
efficiency = [(sp / p) * 100 for sp, p in zip(speedup, processes)]

print("=" * 60)
print("MPI PERFORMANCE ANALYSIS - 10 MILLION ELEMENTS")
print("=" * 60)
print(f"{'Processes':<12} {'Time (s)':<15} {'Speedup':<12} {'Efficiency':<12}")
print("-" * 60)

for p, time, sp, eff in zip(processes, times, speedup, efficiency):
    print(f"{p:<12} {time:<15.3f} {sp:<12.2f} {eff:<12.1f}%")

print("=" * 60)

# Create figure with 2 graphs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ============================================
# GRAPH 1: Processes vs Execution Time (LINE GRAPH)
# ============================================
ax1.plot(processes, times, 'go-', linewidth=3, markersize=12, 
         markerfacecolor='white', markeredgewidth=2.5, marker='o',
         markeredgecolor='green')

ax1.set_xlabel('Number of MPI Processes', fontsize=12, fontweight='bold')
ax1.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
ax1.set_title('MPI: Processes vs Execution Time\nVector Size: 10,000,000', 
              fontsize=14, fontweight='bold', pad=20)

ax1.set_xticks(processes)
ax1.grid(True, alpha=0.3, linestyle='--')

# Add value labels with exact coordinates
for p, time in zip(processes, times):
    ax1.text(p, time + 0.002, f'{time:.3f}s', 
             ha='center', va='bottom', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.9))

# Highlight best performance point
best_idx = times.index(min(times))
ax1.plot(processes[best_idx], times[best_idx], 'r*', markersize=25, 
         markeredgewidth=2, markeredgecolor='darkred',
         label=f'Best: {processes[best_idx]} procs ({times[best_idx]:.3f}s)')
ax1.legend(loc='upper right', fontsize=11)

# Add horizontal reference line for serial time
ax1.axhline(y=times[0], color='red', linestyle=':', alpha=0.5, linewidth=1)
ax1.text(4.5, times[0] + 0.001, f'Serial baseline: {times[0]:.3f}s',
         color='red', fontsize=10, alpha=0.8, fontstyle='italic')

# ============================================
# GRAPH 2: Processes vs Speedup (LINE GRAPH)
# ============================================
ax2.plot(processes, speedup, 'mo-', linewidth=3, markersize=12,
         markerfacecolor='white', markeredgewidth=2.5, marker='s',
         markeredgecolor='purple', label='Actual Speedup')

# Ideal speedup line (for comparison)
ax2.plot(processes, processes, 'k--', linewidth=2, alpha=0.6, label='Ideal Speedup')

ax2.set_xlabel('Number of MPI Processes', fontsize=12, fontweight='bold')
ax2.set_ylabel('Speedup (Relative to 1 process)', fontsize=12, fontweight='bold')
ax2.set_title('MPI: Processes vs Speedup\nBaseline: 1 process (0.038s)', 
              fontsize=14, fontweight='bold', pad=20)

ax2.set_xticks(processes)
ax2.legend(fontsize=11, loc='upper left')
ax2.grid(True, alpha=0.3, linestyle='--')

# Add speedup value labels at exact coordinates
for p, sp in zip(processes, speedup):
    ax2.text(p, sp + 0.08, f'{sp:.2f}x', 
             ha='center', va='bottom', fontsize=11, fontweight='bold', color='purple',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lavender', alpha=0.9,
                      edgecolor='purple'))

# ============================================
# REARRANGED TEXT BOXES - MOVED PERFORMANCE ANALYSIS UP
# ============================================

# Performance Analysis Box - MOVED TO TOP RIGHT (under title)
analysis = "Performance Analysis:\n"
analysis += f"• Optimal: {processes[best_idx]} processes ({times[best_idx]:.3f}s)\n"
analysis += f"• Max speedup: {max(speedup):.2f}x\n"
analysis += f"• Communication overhead dominates at 4+ processes\n"
analysis += f"• Problem size too small for effective MPI scaling"

ax2.text(0.98, 0.90, analysis, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='lightblue', 
                  alpha=0.9, edgecolor='navy'))

# Efficiency Box - MOVED TO TOP LEFT
efficiency_text = "Parallel Efficiency:\n" + "\n".join([f"{p} procs: {eff:.1f}%" 
                                              for p, eff in zip(processes, efficiency)])

ax2.text(0.02, 0.90, efficiency_text, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='lightyellow', 
                  alpha=0.9, edgecolor='gold'))

# Degraded Performance Indicator - at bottom
if len(processes) > best_idx + 1:
    for i in range(best_idx + 1, len(processes)):
        ax2.plot(processes[i], speedup[i], 'rx', markersize=15, markeredgewidth=2)
        ax2.text(processes[i], speedup[i] - 0.15, 'Degraded', 
                 ha='center', color='red', fontsize=9, fontstyle='italic')

# Add a note at the bottom
ax2.text(0.5, 0.02, "MPI communication overhead > computation for small problems",
         transform=ax2.transAxes, ha='center', fontsize=9, fontstyle='italic',
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))

# Adjust layout and save
plt.tight_layout()
plt.savefig('mpi_performance_line_graphs.png', dpi=300, bbox_inches='tight')
plt.savefig('mpi_performance_line_graphs.pdf', bbox_inches='tight')

print("\n✅ MPI Line Graphs created successfully!")
print("📁 Files saved:")
print("   - mpi_performance_line_graphs.png")
print("   - mpi_performance_line_graphs.pdf")
print()

# Show the plot
plt.show()

# Save data to file
with open('mpi_performance_data_10M.txt', 'w') as f:
    f.write("MPI PERFORMANCE DATA - 10 MILLION ELEMENTS\n")
    f.write("=" * 60 + "\n")
    f.write(f"{'Processes':<12} {'Time (s)':<15} {'Speedup':<12} {'Efficiency':<12}\n")
    f.write("-" * 60 + "\n")
    for p, time, sp, eff in zip(processes, times, speedup, efficiency):
        f.write(f"{p:<12} {time:<15.3f} {sp:<12.2f} {eff:<12.1f}%\n")
    
    f.write("\n" + "=" * 60 + "\n")
    f.write("OBSERVATIONS:\n")
    f.write(f"- Vector Size: 10,000,000 elements\n")
    f.write(f"- Memory usage: ~{(10000000 * 3 * 4) / (1024**2):.1f} MB total\n")
    f.write(f"- Best performance at {processes[best_idx]} processes\n")
    f.write(f"- Maximum speedup: {max(speedup):.2f}x\n")
    f.write("- Performance decreases at 4+ processes due to communication overhead\n")
    f.write("- MPI shows poor scaling for this small problem size\n")

print("📄 Data file created: mpi_performance_data_10M.txt")
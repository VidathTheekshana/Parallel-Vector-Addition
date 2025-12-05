
import matplotlib.pyplot as plt
import numpy as np

print("=" * 80)
print("COMPARATIVE PERFORMANCE ANALYSIS - ALL IMPLEMENTATIONS")
print("Vector Addition: 10,000,000 elements")
print("=" * 80)

# ============================================
# ENTER YOUR ACTUAL TEST RESULTS HERE:
# ============================================

# OpenMP Results (from your tests - 10M elements)
print("\nEnter OpenMP Results (10M elements):")
omp_threads = [1, 2, 4, 8, 16]
omp_times = []
for t in omp_threads:
    time = float(input(f"  Time for {t} threads (seconds): "))
    omp_times.append(time)

# MPI Results (10M elements - from updated test)
print("\nEnter MPI Results (10M elements):")
mpi_processes = [1, 2, 4, 8]
mpi_times = []
for p in mpi_processes:
    time = float(input(f"  Time for {p} processes (seconds): "))
    mpi_times.append(time)

# CUDA Results (10M elements)
print("\nEnter CUDA Results (10M elements):")
cuda_threads = [32, 64, 128, 256, 512]
cuda_times = []
for t in cuda_threads:
    time = float(input(f"  Time for {t} threads/block (seconds): "))
    cuda_times.append(time)

# ============================================
# CALCULATIONS
# ============================================
# Serial baseline (OpenMP with 1 thread)
serial_time = omp_times[0]

# Speedup calculations
omp_speedup = [serial_time / t for t in omp_times]
mpi_speedup = [serial_time / t for t in mpi_times]
cuda_speedup = [serial_time / t for t in cuda_times]

# Best performance for each
omp_best_idx = omp_times.index(min(omp_times))
mpi_best_idx = mpi_times.index(min(mpi_times))
cuda_best_idx = cuda_times.index(min(cuda_times))

omp_best_time = omp_times[omp_best_idx]
mpi_best_time = mpi_times[mpi_best_idx]
cuda_best_time = cuda_times[cuda_best_idx]

print("\n" + "=" * 80)
print("PERFORMANCE SUMMARY")
print("=" * 80)
print(f"{'Technology':<12} {'Best Config':<15} {'Time (s)':<12} {'Speedup vs Serial':<15}")
print("-" * 80)
print(f"{'OpenMP':<12} {omp_threads[omp_best_idx]} threads{'':<8} {omp_best_time:<12.3f} {serial_time/omp_best_time:<15.2f}x")
print(f"{'MPI':<12} {mpi_processes[mpi_best_idx]} processes{'':<4} {mpi_best_time:<12.3f} {serial_time/mpi_best_time:<15.2f}x")
print(f"{'CUDA':<12} {cuda_threads[cuda_best_idx]} thr/blk{'':<5} {cuda_best_time:<12.3f} {serial_time/cuda_best_time:<15.2f}x")
print("-" * 80)

# ============================================
# CREATE COMPARATIVE GRAPHS
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# GRAPH 1: Execution Time Trends
ax1 = axes[0, 0]

# Normalize x-axis for comparison
x_omp = np.arange(len(omp_threads))
x_mpi = np.arange(len(mpi_processes)) + 0.2
x_cuda = np.arange(len(cuda_threads)) + 0.4

ax1.plot(x_omp, omp_times, 'bo-', linewidth=3, markersize=10, 
         markerfacecolor='white', markeredgewidth=2, label='OpenMP')
ax1.plot(x_mpi, mpi_times, 'go-', linewidth=3, markersize=10,
         markerfacecolor='white', markeredgewidth=2, label='MPI')
ax1.plot(x_cuda, cuda_times, 'ro-', linewidth=3, markersize=10,
         markerfacecolor='white', markeredgewidth=2, label='CUDA')

ax1.set_xlabel('Configuration (normalized scale)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
ax1.set_title('Execution Time Comparison\nAll Implementations - 10 Million Elements', 
              fontsize=14, fontweight='bold', pad=20)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xticks([])

# Add value labels
for x, t, tech in zip([x_omp, x_mpi, x_cuda], [omp_times, mpi_times, cuda_times], ['O', 'M', 'C']):
    for xi, ti in zip(x, t):
        ax1.text(xi, ti + max(t)*0.02, f'{ti:.3f}s', 
                ha='center', va='bottom', fontsize=9)

# GRAPH 2: Speedup Comparison
ax2 = axes[0, 1]

ax2.plot(x_omp, omp_speedup, 'bo-', linewidth=3, markersize=10,
         markerfacecolor='white', markeredgewidth=2, label='OpenMP Speedup')
ax2.plot(x_mpi, mpi_speedup, 'go-', linewidth=3, markersize=10,
         markerfacecolor='white', markeredgewidth=2, label='MPI Speedup')
ax2.plot(x_cuda, cuda_speedup, 'ro-', linewidth=3, markersize=10,
         markerfacecolor='white', markeredgewidth=2, label='CUDA Speedup')

# Ideal speedup line
max_x = max(len(omp_threads), len(mpi_processes), len(cuda_threads))
ideal_x = np.arange(max_x)
ideal_y = [i+1 for i in ideal_x]
ax2.plot(ideal_x, ideal_y, 'k--', alpha=0.5, linewidth=2, label='Ideal Speedup')

ax2.set_xlabel('Configuration (normalized scale)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Speedup (vs Serial: 1 thread)', fontsize=12, fontweight='bold')
ax2.set_title('Speedup Comparison\nRelative to Serial Implementation', 
              fontsize=14, fontweight='bold', pad=20)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xticks([])

# GRAPH 3: Best Performance Bar Chart
ax3 = axes[1, 0]

technologies = ['OpenMP', 'MPI', 'CUDA']
best_times = [omp_best_time, mpi_best_time, cuda_best_time]
best_configs = [
    f'{omp_threads[omp_best_idx]} threads',
    f'{mpi_processes[mpi_best_idx]} processes', 
    f'{cuda_threads[cuda_best_idx]} thr/blk'
]
colors = ['blue', 'green', 'red']

bars = ax3.bar(technologies, best_times, color=colors, alpha=0.8, 
               edgecolor='black', linewidth=2)

ax3.set_xlabel('Implementation', fontsize=12, fontweight='bold')
ax3.set_ylabel('Best Execution Time (seconds)', fontsize=12, fontweight='bold')
ax3.set_title('Best Performance Achieved\nOptimal Configuration for Each Technology', 
              fontsize=14, fontweight='bold', pad=20)
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, time, config in zip(bars, best_times, best_configs):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + max(best_times)*0.02,
             f'{time:.3f}s\n({config})', ha='center', va='bottom', 
             fontsize=10, fontweight='bold')

# GRAPH 4: Speedup vs Serial (Bar Chart)
ax4 = axes[1, 1]

speedup_vs_serial = [
    serial_time/omp_best_time,
    serial_time/mpi_best_time,
    serial_time/cuda_best_time
]

bars2 = ax4.bar(technologies, speedup_vs_serial, color=colors, alpha=0.8,
                edgecolor='black', linewidth=2)

ax4.set_xlabel('Implementation', fontsize=12, fontweight='bold')
ax4.set_ylabel('Speedup vs Serial (times faster)', fontsize=12, fontweight='bold')
ax4.set_title('Overall Speedup Comparison\nHow much faster than serial execution', 
              fontsize=14, fontweight='bold', pad=20)
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, speedup_val in zip(bars2, speedup_vs_serial):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{speedup_val:.1f}x', ha='center', va='bottom', 
             fontsize=12, fontweight='bold')

# Add horizontal lines
ax4.axhline(y=1, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Serial (1x)')
ax4.legend(fontsize=11)

plt.tight_layout()
plt.savefig('comparative_performance_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('comparative_performance_analysis.pdf', bbox_inches='tight')

print(f"\n✅ Comparative graphs saved:")
print("   - comparative_performance_analysis.png")
print("   - comparative_performance_analysis.pdf")

# ============================================
# CREATE ANALYSIS REPORT
# ============================================
with open('comparative_analysis_complete.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("COMPARATIVE PARALLEL COMPUTING ANALYSIS\n")
    f.write("Vector Addition Algorithm - 10 Million Elements\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("TEST RESULTS\n")
    f.write("-" * 40 + "\n")
    
    f.write("OpenMP Performance:\n")
    f.write(f"  Serial (1 thread): {omp_times[0]:.3f}s\n")
    for i, (t, time) in enumerate(zip(omp_threads, omp_times)):
        f.write(f"  {t} threads: {time:.3f}s (Speedup: {omp_speedup[i]:.2f}x)\n")
    
    f.write("\nMPI Performance:\n")
    for i, (p, time) in enumerate(zip(mpi_processes, mpi_times)):
        f.write(f"  {p} processes: {time:.3f}s (Speedup: {mpi_speedup[i]:.2f}x)\n")
    
    f.write("\nCUDA Performance:\n")
    for i, (t, time) in enumerate(zip(cuda_threads, cuda_times)):
        f.write(f"  {t} threads/block: {time:.3f}s (Speedup: {cuda_speedup[i]:.2f}x)\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("PERFORMANCE RANKING\n")
    f.write("-" * 40 + "\n")
    
    # Sort by performance
    tech_data = [
        ("CUDA", cuda_best_time, cuda_threads[cuda_best_idx]),
        ("OpenMP", omp_best_time, omp_threads[omp_best_idx]),
        ("MPI", mpi_best_time, mpi_processes[mpi_best_idx])
    ]
    tech_data.sort(key=lambda x: x[1])  # Sort by time
    
    f.write("1. {}: {:.3f}s ({} config)\n".format(*tech_data[0]))
    f.write("2. {}: {:.3f}s ({} config)\n".format(*tech_data[1]))
    f.write("3. {}: {:.3f}s ({} config)\n".format(*tech_data[2]))
    
    f.write(f"\nPerformance Ratio (Fastest to Slowest): 1 : {tech_data[1][1]/tech_data[0][1]:.1f} : {tech_data[2][1]/tech_data[0][1]:.1f}\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("RECOMMENDATION WITH SUFFICIENT RESOURCES\n")
    f.write("-" * 40 + "\n")
    
    fastest_tech = tech_data[0][0]
    f.write(f"Based on the analysis, {fastest_tech} demonstrates the best performance\n")
    f.write("for the vector addition algorithm with 10 million elements.\n\n")
    
    f.write("RECOMMENDED APPROACH:\n")
    f.write("1. PRIMARY: Use {} implementation\n".format(fastest_tech))
    f.write("   - Provides {:.1f}x speedup over serial\n".format(serial_time/tech_data[0][1]))
    f.write("   - Optimal configuration: {}\n".format(tech_data[0][2]))
    
    f.write("\n2. HYBRID STRATEGY (if resources allow):\n")
    f.write("   a. Use {} for main computation\n".format(tech_data[0][0]))
    f.write("   b. Use {} for pre/post-processing\n".format(tech_data[1][0]))
    f.write("   c. Use {} for data distribution across nodes\n".format(tech_data[2][0]))
    
    f.write("\n3. SCALABILITY CONSIDERATIONS:\n")
    f.write("   - For larger datasets (>100M): Consider MPI for distributed memory\n")
    f.write("   - For real-time processing: CUDA provides lowest latency\n")
    f.write("   - For CPU-only systems: OpenMP offers best performance\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("STRENGTHS AND WEAKNESSES ANALYSIS\n")
    f.write("-" * 40 + "\n")
    
    f.write("OPENMP:\n")
    f.write("  ✓ Simple implementation with pragmas\n")
    f.write("  ✓ Excellent for shared-memory systems\n")
    f.write("  ✗ Limited to single machine\n")
    f.write("  ✗ Cannot utilize GPU acceleration\n\n")
    
    f.write("MPI:\n")
    f.write("  ✓ Scales across multiple machines\n")
    f.write("  ✓ Handles very large distributed datasets\n")
    f.write("  ✗ Higher programming complexity\n")
    f.write("  ✗ Communication overhead significant\n\n")
    
    f.write("CUDA:\n")
    f.write("  ✓ Massive parallelism (1000s of threads)\n")
    f.write("  ✓ Highest performance for data-parallel tasks\n")
    f.write("  ✗ Requires NVIDIA GPU hardware\n")
    f.write("  ✗ Complex memory management\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("CONCLUSION\n")
    f.write("-" * 40 + "\n")
    f.write("For vector addition with 10 million elements:\n")
    f.write(f"• {fastest_tech} is the fastest implementation\n")
    f.write("• The choice depends on available hardware and problem scale\n")
    f.write("• Hybrid approaches provide maximum performance when resources are available\n")
    f.write("• Consider problem size, hardware constraints, and development complexity\n")

print("\n✅ Comparative analysis report saved: comparative_analysis_complete.txt")

plt.show()

# ============================================
# CREATE DATA TABLE FOR ASSIGNMENT
# ============================================
print("\n" + "=" * 80)
print("DATA FOR YOUR ASSIGNMENT REPORT")
print("=" * 80)

print("\n1. EXECUTION TIMES (seconds):")
print("-" * 40)
print(f"{'Config':<15} {'OpenMP':<10} {'MPI':<10} {'CUDA':<10}")
print("-" * 40)

max_configs = max(len(omp_threads), len(mpi_processes), len(cuda_threads))
for i in range(max_configs):
    omp = f"{omp_times[i]:.3f}" if i < len(omp_times) else "-"
    mpi = f"{mpi_times[i]:.3f}" if i < len(mpi_times) else "-"
    cuda = f"{cuda_times[i]:.3f}" if i < len(cuda_times) else "-"
    
    if i < len(omp_threads):
        config = f"{omp_threads[i]} threads"
    elif i < len(mpi_processes):
        config = f"{mpi_processes[i]} processes"
    else:
        config = f"{cuda_threads[i]} thr/blk"
    
    print(f"{config:<15} {omp:<10} {mpi:<10} {cuda:<10}")

print("\n2. SPEEDUP VS SERIAL (times faster):")
print("-" * 40)
print(f"{'Config':<15} {'OpenMP':<10} {'MPI':<10} {'CUDA':<10}")
print("-" * 40)

for i in range(max_configs):
    omp = f"{omp_speedup[i]:.2f}x" if i < len(omp_speedup) else "-"
    mpi = f"{mpi_speedup[i]:.2f}x" if i < len(mpi_speedup) else "-"
    cuda = f"{cuda_speedup[i]:.2f}x" if i < len(cuda_speedup) else "-"
    
    if i < len(omp_threads):
        config = f"{omp_threads[i]} threads"
    elif i < len(mpi_processes):
        config = f"{mpi_processes[i]} processes"
    else:
        config = f"{cuda_threads[i]} thr/blk"
    
    print(f"{config:<15} {omp:<10} {mpi:<10} {cuda:<10}")

print("\n" + "=" * 80)
print("COMPARATIVE ANALYSIS COMPLETE!")
print("=" * 80)
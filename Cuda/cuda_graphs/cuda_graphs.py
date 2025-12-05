import subprocess
import numpy as np

thread_configs = [32, 64, 128, 256, 512, 1024]
times = []

print("Testing compute-intensive kernel...")
for threads in thread_configs:
    print(f"\n{'='*50}")
    print(f"Testing {threads} threads per block")
    print('='*50)
    
    # Run 3 times and take average
    run_times = []
    for run in range(3):
        result = subprocess.run(["./compute_intensive", str(threads)], 
                               capture_output=True, text=True)
        
        # Extract time
        for line in result.stdout.split('\n'):
            if line.startswith("Time:"):
                time_val = float(line.split()[1])
                run_times.append(time_val)
                break
    
    avg_time = np.mean(run_times) if run_times else 0
    times.append(avg_time)
    print(f"Average time: {avg_time:.6f} seconds")

# Create better graphs
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(thread_configs, times, 'bo-', linewidth=3, markersize=10, markerfacecolor='red')
plt.xlabel('Threads per Block', fontsize=12, fontweight='bold')
plt.ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
plt.title('CUDA Performance: Threads/Block vs Time\n(Compute-Intensive Kernel)', 
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(thread_configs)

plt.subplot(1, 2, 2)
speedup = [times[0]/t for t in times]
plt.plot(thread_configs, speedup, 'ro-', linewidth=3, markersize=10, markerfacecolor='blue')
plt.xlabel('Threads per Block', fontsize=12, fontweight='bold')
plt.ylabel('Speedup (relative to 32 threads)', fontsize=12, fontweight='bold')
plt.title('CUDA Performance: Threads/Block vs Speedup\n(Baseline: 32 threads)', 
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
plt.xticks(thread_configs)

plt.tight_layout()
plt.show()

# Print results table
print("\n" + "="*60)
print("SUMMARY OF RESULTS")
print("="*60)
print(f"{'Threads/Block':<15} {'Time (s)':<15} {'Speedup':<15}")
print("-"*45)
for i, (threads, time_val) in enumerate(zip(thread_configs, times)):
    speedup_val = times[0]/time_val if i > 0 else 1.0
    print(f"{threads:<15} {time_val:<15.6f} {speedup_val:<15.2f}")
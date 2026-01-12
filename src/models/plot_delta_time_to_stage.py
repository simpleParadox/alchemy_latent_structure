import numpy as np
import matplotlib.pyplot as plt

# Single-layer freezing
layers = [0, 1, 2, 3, 4]
delta_t_single = [np.inf, 100, 15, 5, 0]  # example values

# Optional block-wise freezing
block_labels = ['{0,1}', '{0,1,2}', '{2,3}', '{2,3,4}']
delta_t_blocks = [np.inf, 180, 40, 20]

epsilon = 10  # tolerance threshold

fig, ax = plt.subplots(figsize=(7, 4))

# Plot single-layer curves
ax.plot(
    layers,
    delta_t_single,
    marker='o',
    linestyle='-',
    label='Single-layer freeze'
)

# Plot block-wise freezes (offset x-axis to avoid overlap)
block_x = np.arange(len(block_labels)) + 0.15
ax.plot(
    block_x,
    delta_t_blocks,
    marker='s',
    linestyle='--',
    label='Block freeze'
)

# Epsilon line
ax.axhline(
    epsilon,
    color='gray',
    linestyle=':',
    linewidth=1,
    label=r'$\epsilon$ threshold'
)

# Handle infinite delays explicitly
for x, y in zip(layers, delta_t_single):
    if not np.isfinite(y):
        ax.text(x, ax.get_ylim()[1] * 0.9, '∞', ha='center')

# Axis formatting
ax.set_xlabel('Frozen layer / block')
ax.set_ylabel(r'$\Delta t$ (epochs)')
ax.set_title('Delay in terminal stage after freezing')
ax.legend(frameon=False)

# Optional: log-scale if delays vary widely
# ax.set_yscale('log')

plt.tight_layout()
plt.show()
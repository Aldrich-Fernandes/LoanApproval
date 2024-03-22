import matplotlib.pyplot as plt
import numpy as np

# True labels
true_values = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0] 

# Output values
output_values = [0.7084669482800612, 0.5879665328499811, 0.7086921485543072, 0.6291691037334843, 0.7049169249142209, 0.10838592859735935, 0.5330615461461843, 0.06833316000915629, 0.5619094555765303, 0.12883961127849283, 0.7070483832353431, 0.6609273202780119, 0.10391419889262296, 0.4858769472369085, 0.5685559697643381, 0.5397965552523867, 0.6498467844880464, 0.6034035183978932, 0.538714099125489, 0.6785427486448159, 0.6019057851543312, 0.06815637485116365, 0.7087720730170183, 0.6415085972923438, 0.06926112181403346, 0.1089379557055991, 0.8188981327201181, 0.6280038021224638, 0.060421354457962484, 0.6652573506546808, 0.7022296978928594, 0.6946366848631524, 0.52917748630537, 0.09781890909340321, 0.6111272288560543, 0.6947178995469252, 0.6631016001003652, 0.6154254409563523, 0.668656588509149, 0.6618877661105579, 0.7355876792464741, 0.6389705087953542, 0.7114353119579283, 0.10645252102913891, 0.11589301003695748, 0.523387845669281, 0.5269693689614154, 0.6655099813877081, 0.44548723842172844, 0.4409874815662892]

# Convert true and output values to numpy arrays
true_values = np.array(true_values)
output_values = np.array(output_values)

# Calculate difference between true and output values
diff_values = np.abs(true_values - output_values)

# Create subplot for scatter plot
plt.figure(figsize=(14, 6))

# Subplot for scatter plot
plt.subplot(1, 2, 1)
plt.scatter(true_values, output_values, color='blue', alpha=0.7)
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('Scatter graph of True vs. Predicted values')
plt.grid(True)

# Subplot for heatmap
plt.subplot(1, 2, 2)
plt.hist2d(true_values, diff_values, bins=(20, 20), cmap=plt.cm.Blues)
plt.colorbar(label='Frequency')
plt.xlabel('True values')
plt.ylabel('Absolute Difference (Predicted - True)')
plt.title('Heatmap of True vs. Absolute Difference')

plt.tight_layout()
plt.show()
import torch
import torch.nn as nn
import torch.nn.functional as F

tensor1 = torch.tensor([1.0,2.0,3.0])
tensor2 = torch.tensor([4.0,5.0,6.0])

# Define the Loss Function
def calculate_loss(predicted_displacement, true_displacement):
    # Calculate the displacement error
    displacement_error = predicted_displacement - true_displacement
    # Calculate the average displacement error
    ADE = torch.mean(displacement_error)
    return ADE

# Calculate the average displacement error during training or evaluation
ADE = calculate_loss(tensor1, tensor2)

# Print the average displacement error
print(f"Average Displacement Error: {ADE}")
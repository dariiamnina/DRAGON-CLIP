import torch
import torch.nn as nn

import torch
import torch.nn as nn

class SelfAttentionLayer(nn.Module):
    def __init__(self, input_size1, input_size2):
        super(SelfAttentionLayer, self).__init__()

        self.input_size1 = input_size1
        self.input_size2 = input_size2
        
        # Linear layers for query and key projections
        self.query_linear = nn.Linear(input_size1, input_size1)
        self.key_linear = nn.Linear(input_size2, input_size1)


    def forward(self, input):
        # Split the input into the two vectors
        vector1, vector2 = input[:, :self.input_size1], input[:, self.input_size1:]
        
        # Add a gradient hook to ensure the gradient flows through vector1
        vector1.register_hook(lambda grad: grad)
        
        # Project the query and key vectors
        query = self.query_linear(vector1)
        key = self.key_linear(vector2)
        
        # Calculate attention scores using dot product
        attention_scores = torch.matmul(query, key.transpose(0, 1))
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # Compute the weighted sum of the first vector using the attention weights
        # output = torch.matmul(attention_weights, vector1.unsqueeze(0))
        # output = output.squeeze(0)
        output = torch.matmul(attention_weights, vector1.unsqueeze(0))
        output = output.view(output.size(1), -1)
        
        return output


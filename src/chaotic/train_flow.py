import torch
import torch.nn as nn
import torch.optim as optim

class ConditionalNormalizingFlow(nn.Module):
    def __init__(self, base_distribution, flow_layers):
        super(ConditionalNormalizingFlow, self).__init__()
        self.base_distribution = base_distribution
        self.flow_layers = flow_layers

    def forward(self, x, conditions):
        log_prob = self.base_distribution.log_prob(x)
        for layer in self.flow_layers:
            x, log_det_jacobian = layer(x, conditions)
            log_prob += log_det_jacobian
        return x, log_prob

    def inverse(self, z, conditions):
        for layer in reversed(self.flow_layers):
            z, log_det_jacobian = layer.inverse(z, conditions)
        return z

def train_flow(model, data_loader, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for x, conditions in data_loader:
            optimizer.zero_grad()
            z, log_prob = model(x, conditions)
            loss = -log_prob.mean()
            loss.backward()
            optimizer.step()

# Example usage
if __name__ == "__main__":
    # Define base distribution and flow layers
    base_distribution = torch.distributions.Normal(torch.zeros(2), torch.ones(2))
    flow_layers = []  # Add flow layers here

    model = ConditionalNormalizingFlow(base_distribution, flow_layers)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Assuming data_loader is defined
    # train_flow(model, data_loader, optimizer, num_epochs=100)
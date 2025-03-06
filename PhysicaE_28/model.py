import torch 
from torch import nn 

class PINN(nn.Module):
    def __init__(self, n_hidden: int, n_layers:int):
        super(PINN, self).__init__()
        self.input_layer = nn.Linear(2, n_hidden) # inputs: (rho, z)
        self.hidden_layers = nn.ModuleList([
             nn.Linear(n_hidden, n_hidden) for _ in range(n_layers)
        ])
        self.output_layer = nn.Linear(n_hidden, 1) # output: Ψ(rho,z) (real)
        self.activation = nn.Tanh()

        self.energy = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        # self._init_weights() #till off 
    
    def _init_weights(self):
        """
        Xavier or other suitable initialization
        """
        for layer in [self.input_layer, *self.hidden_layers, self.output_layer]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, rho, z):
        x_input= torch.cat((rho, z), dim=1) #shape (batch, 2)

        x = self.activation(self.input_layer(x_input))
        for hl in self.hidden_layers:
            x = self.activation(hl(x))
        psi = self.output_layer(x)

        return psi
    
    def get_energy(self):
        return self.energy
    

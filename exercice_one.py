import torch
import torch.nn as nn

class my_model(nn.Module) :
    def __init__(self, inputs=4, layer=32, outputs=2):
        super().__init__()
        self.net = nn.Sequential(         
            nn.Linear(inputs, layer),
            nn.ReLU(),
            nn.Linear(layer, outputs)
        )


    def forward(self, x):
        return self.net(x)

torch.manual_seed(7)
model = my_model()
x_tensor = torch.rand(1, 4)
print(x_tensor,"\n\n")
new_x = model(x_tensor)

print(new_x, new_x.shape)
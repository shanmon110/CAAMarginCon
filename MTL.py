class MTL(nn.Module):

        
    def forward(self, optim, features, nloss1, nloss2):
           features.register_hook(save_grad('Z'))
           nloss1.backward(retain_graph=True)
           theta1 = grads['Z'].reshape(-1)          
           optim.zero_grad()           
           nloss2.backward(retain_graph=True)
           theta2 = grads['Z'].reshape(-1)         
           part1 = torch.matmul((theta2 - theta1), theta2.T)          
           part2 = torch.norm(theta1 - theta2, p=2)          
           part2.pow(2)
           alpha = torch.div(part1, part2)
           min = torch.ones_like(alpha)
           alpha = torch.where(alpha > 1, min, alpha)
           min = torch.zeros_like(alpha)
           alpha = torch.where(alpha < 0, min, alpha)        
           alpha1 = alpha
           alpha2 = (1 - alpha)
           optim.zero_grad()
           loss = alpha1 * nloss1 + alpha2 * nloss2
           
        return loss

    grads = {}
    def save_grad(name):
        def hook(grad):
            grads[name] = grad
        return hook

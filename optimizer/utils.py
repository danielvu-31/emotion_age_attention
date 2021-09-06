from torch.optim import Adam
from pcgrad import PCGrad
from sam import SAM

def load_optimizer(model, lr, optimizer_name, weight_decay=1e-6):
    if optimizer_name == "sam":
        optimizer = SAM(filter(lambda p: p.requires_grad, model.parameters()),
                        Adam, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "pcgrad":
        optimizer = PCGrad(Adam(model.parameters()))
    else:
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=lr, weight_decay=weight_decay)
    return optimizer


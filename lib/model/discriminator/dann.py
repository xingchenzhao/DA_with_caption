import torch
import torchvision.models as torchmodels


class DANN(torch.nn.Module):
    def __init__(self, num_classes):

        super().__init__()
        self.init_layer = torch.nn.MaxPool2d(2)
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.layers = torch.nn.Sequential(  #torch.nn.Linear(2560, 2048),
            #torch.nn.ReLU(), torch.nn.Dropout(),
            torch.nn.Linear(1280, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(512, num_classes))

        # disc head get's default initialization

    def forward(self, x):
        x = self.init_layer(x)
        x = self.flatten(x)
        return self.layers(x)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta, reverse=True):
        ctx.beta = beta
        ctx.reverse = reverse
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.reverse:
            # print('adv example reversed')
            return (grad_output * -ctx.beta), None, None
        else:
            # print('adv example not reversed')
            return (grad_output * ctx.beta), None, None


def grad_reverse(x, beta=1.0, reverse=True):
    return GradReverse.apply(x, beta, reverse)


class Discriminator(torch.nn.Module):
    def __init__(self, head, reverse=True):

        super(Discriminator, self).__init__()
        self.head = head
        self.beta = 0.0
        self.reverse = reverse

    def set_beta(self, beta):
        self.beta = beta

    def forward(self, x, use_grad_reverse=True):
        if use_grad_reverse:
            x = grad_reverse(x, self.beta, reverse=self.reverse)
        x = self.head(x)
        return x
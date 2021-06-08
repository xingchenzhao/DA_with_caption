import torch
import torch.nn.functional as F


def guassian_kernel(source,
                    target,
                    kernel_mul=2.0,
                    kernel_num=5,
                    fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    source = torch.flatten(source, start_dim=1)
    target = torch.flatten(target, start_dim=1)
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)),
                                       int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)),
                                       int(total.size(1)))
    L2_distance = ((total0 - total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
    bandwidth /= kernel_mul**(kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [
        torch.exp(-L2_distance / bandwidth_temp)
        for bandwidth_temp in bandwidth_list
    ]
    return sum(kernel_val)  #/len(kernel_val)


def mmd_rbf_accelerate(source,
                       target,
                       kernel_mul=2.0,
                       kernel_num=5,
                       fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source,
                              target,
                              kernel_mul=kernel_mul,
                              kernel_num=kernel_num,
                              fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i + 1) % batch_size
        t1, t2 = s1 + batch_size, s2 + batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)


def mmd_rbf_noaccelerate(source,
                         target,
                         kernel_mul=2.0,
                         kernel_num=5,
                         fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source,
                              target,
                              kernel_mul=kernel_mul,
                              kernel_num=kernel_num,
                              fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]

    xx_loss = torch.mean(XX)
    yy_loss = torch.mean(YY)
    xy_loss = torch.mean(XY)
    yx_loss = torch.mean(YX)
    loss = xx_loss + yy_loss - xy_loss - yx_loss
    return loss


def CORAL_loss(source, target):
    source = torch.flatten(source, start_dim=1)
    target = torch.flatten(target, start_dim=1)
    d = source.data.shape[1]
    ns, nt = source.data.shape[0], target.data.shape[0]
    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm / (ns - 1)

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt / (nt - 1)

    # frobenius norm between source and target
    loss = torch.mul((xc - xct), (xc - xct))
    loss = torch.sum(loss) / (4 * d * d)
    return loss

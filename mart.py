import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from loss.loss import MaximalCodingRateReduction


def mart_loss(mcrAt,
              n_classes,
              epsMCR2,
              mcrBeta,
              model,
              x_natural,
              y,
              optimizer,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=6.0,
              distance='l_inf'):
    kl = nn.KLDivLoss(reduction='none')
    # MCR2 Loss
    criterion1 = MaximalCodingRateReduction(eps=epsMCR2)
    
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    # x_adv = x_natural.detach() + torch.empty_like(x_natural).uniform_(epsilon, epsilon)
    # x_adv = torch.clamp(x_adv, min=0, max=1).detach()
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    
    # zero gradient
    optimizer.zero_grad()

    logits = model(x_natural)

    logits_adv = model(x_adv)

    adv_probs = F.softmax(logits_adv, dim=1)

    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]

    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

    loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

    nat_probs = F.softmax(logits, dim=1)

    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
    loss2 = loss_adv + float(beta) * loss_robust
    
    adv_outputs, outF = model(x_adv, 1)
    if mcrAt == 1:
        loss1, _, _ = criterion1(outF, y, num_classes=n_classes)
        k = mcrBeta
        loss =  loss2 + k * loss1
    else:
        loss = loss2

    return loss, adv_outputs

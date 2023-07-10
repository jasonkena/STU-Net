import torch
import numpy as np

def remap_output(output, labels, zero_could_be, used_labels):
    # output: torch.Size([2, 11, 128, 128, 128])
    groups = []
    used_labels = sorted(used_labels)
    for x in used_labels:
        if x == 0:
            groups.append(torch.max(output[:, zero_could_be], 1, keepdim=True)[0])
        else:
            if "/" in labels[str(x)]:
                array = []
                for y in labels[str(x)].split("/"):
                    y = int(y)
                    array.append(output[:, y:y+1])
                groups.append(torch.max(torch.cat(array, dim=1), 1, keepdim=True)[0])
            else:
                groups.append(output[:, x:x+1])


    output = torch.cat(groups, dim=1)
    assert output.shape[1] == len(used_labels)

    return output
    
def remap_target(target, used_labels):
    # target: torch.Size([2, 1, 128, 128, 128])
    used_labels = sorted(used_labels)
    mapping = -torch.ones(max(used_labels)+1, dtype=torch.long).to(target.device)
    for i, x in enumerate(used_labels):
        mapping[x] = i
    
    shape = target.shape
    target = mapping[target.view(-1).long()].view(shape)
    assert target.min() >= 0

    return target

def remap_loss(loss, labels, zero_could_be, used_labels):
    # apply the old loss on the hierarchically grouped output
    def inner(output, target):
        # assertions are put in here instead of HeartTrainer, so that inference loading can be done without defining these
        assert labels is not None
        assert zero_could_be is not None
        assert used_labels is not None

        output = remap_output(output, labels, zero_could_be, used_labels)
        target = remap_target(target, used_labels)
        return loss(output, target)

    return inner

def detach_and_convert_numpy(*tensors):
    arrs = [t.detach().cpu().numpy() for t in tensors]
    return arrs


def detach_tensors(*tensors):
    arrs = [t.detach() for t in tensors]
    return arrs

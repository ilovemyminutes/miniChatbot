from torch import nn

def get_l2_distance(x1, x2, dim:int=1):
    return ((x1 - x2)**2).sum(dim=dim)**.5

def get_l1_distance(x1, x2, dim:int=1):
    return ((x1 - x2).abs()).sum(dim=dim)

def get_cosine_similarity(x1, x2, dim: int=1):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    return cos(x1, x2)
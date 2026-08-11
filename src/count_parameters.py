import torch
import torch.nn as nn
from models import OneHiddenLayerClassifier,TwoHiddenLayerClassifier

# --- Parameter counting helpers ---
def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)

def print_param_breakdown(model: nn.Module) -> None:
    total = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        print(f"{name:30s} shape={tuple(p.shape)!s:15s} params={n}")
    print(f"\nTOTAL params: {total}")


# --- Example usage ---
if __name__ == "__main__":
    in_dim = 100
    num_classes = 10

    one = OneHiddenLayerClassifier(in_dim=in_dim, hidden_dim=64, num_classes=num_classes)
    two = TwoHiddenLayerClassifier(in_dim=in_dim, hidden_dim1=64, hidden_dim2=32, num_classes=num_classes)

    print("One hidden layer total:", count_parameters(one))
    print_param_breakdown(one)

    print("\nTwo hidden layers total:", count_parameters(two))
    print_param_breakdown(two)
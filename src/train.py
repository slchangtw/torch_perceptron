import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.perceptron import Perceptron

LEFT_CENTER = (3, 3)
RIGHT_CENTER = (3, -2)
SEED = 42


def get_toy_data(
    batch_size: int,
    left_center: tuple[int, int] = LEFT_CENTER,
    right_center: tuple[int, int] = RIGHT_CENTER,
) -> tuple[torch.tensor, torch.tensor]:
    x_data = []
    y_targets = np.zeros(batch_size)
    for batch_i in range(batch_size):
        if np.random.random() > 0.5:
            x_data.append(np.random.normal(loc=left_center))
        else:
            x_data.append(np.random.normal(loc=right_center))
            y_targets[batch_i] = 1
    x_data = np.stack(x_data)
    return torch.tensor(x_data, dtype=torch.float32), torch.tensor(
        y_targets, dtype=torch.float32
    )


if __name__ == "__main__":
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    input_dim = 2
    lr = 0.001

    batch_size = 1000
    n_epochs = 12
    n_batches = 5

    x_data_static, y_truth_static = get_toy_data(batch_size)

    perceptron = Perceptron(input_dim=input_dim)
    bce_loss = nn.BCELoss()
    optimizer = optim.Adam(params=perceptron.parameters(), lr=lr)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    perceptron.to(device)

    change = 1.0
    last = 10.0
    epsilon = 1e-3
    epoch = 0
    losses = []

    while (change > epsilon) or (epoch < n_epochs) or (last > 0.3):
        for _ in range(n_batches):
            optimizer.zero_grad()
            x_data, y_target = get_toy_data(batch_size)
            x_data = x_data.to(device)
            y_target = y_target.to(device)

            y_pred = perceptron(x_data).squeeze()

            loss = bce_loss(y_pred, y_target)
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            losses.append(loss_value)

            change = abs(last - loss_value)
            last = loss_value
        if epoch % 20 == 0:
            print(f"Epoch: {epoch}, Loss: {loss_value:.4f}, Change: {change:.4f}")
        epoch += 1

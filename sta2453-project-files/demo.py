import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim


class PutNet(nn.Module):
    """
    Example of a Neural Network that could be trained price a put option.
    TODO: modify me!
    """

    def __init__(self) -> None:
        super(PutNet, self).__init__()
        n_layer = 5
        n_hid = 50

        self.inlayer = nn.Linear(5, n_hid)
        self.hid_layers = nn.ModuleList([nn.Linear(n_hid, n_hid) for i in range(n_layer)])
        self.out = nn.Linear(n_hid, 1)

    def forward(self, x):
        l = torch.tensor([0.0, 50.0, 0.0, 0.001, 0.05])
        h = torch.tensor([200.0, 150.0, 5.0, 0.05, 1.5])
        x = (x - l) / (h - l)
        # x = torch.logit(x)

        x = F.relu(self.inlayer(x))
        for i, layer in enumerate(self.hid_layers):
            x = F.relu(self.hid_layers[i](x))

        x = self.out(x)
        return x


def main():
    """Train the model and save the checkpoint"""

    # Create model
    model = PutNet()

    # Load dataset
    df_train = pd.read_csv("training.csv")
    df_val = pd.read_csv("validation.csv")

    # Set up training
    x_train = torch.Tensor(df_train[["S", "K", "T", "r", "sigma"]].to_numpy())
    y_train = torch.Tensor(df_train[["value"]].to_numpy())
    n = x_train.shape[0]

    x_val = torch.Tensor(df_val[["S", "K", "T", "r", "sigma"]].to_numpy())
    y_val = torch.Tensor(df_val[["value"]].to_numpy())

    criterion = nn.MSELoss()
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train for 500 epochs
    for i in range(500):

        # TODO: Modify to account for dataset size
        for batch_i in range(n//1000):
            x_batch = x_train[batch_i*1000:batch_i*1000+1000]
            y_batch = y_train[batch_i*1000:batch_i*1000+1000]
            y_hat = model(x_batch)

            # Calculate training loss
            training_loss = criterion(y_hat, y_batch)

            # Take a step
            optimizer.zero_grad()
            training_loss.backward()
            optimizer.step()

        # train bad examples (100 examples with higher error) especially
        error = (model(x_train) - y_train).abs().squeeze()
        topk, topk_id = torch.topk(error, 100)
        # replicate bad examples (* 10)
        # x_bad = torch.tile(x_batch[topk_id], (2, 1))
        # y_bad = torch.tile(y_batch[topk_id], (2, 1))
        x_bad = x_train[topk_id]
        y_bad = y_train[topk_id]
        y_bad_hat = model(x_bad)

        especial_loss = criterion(y_bad_hat, y_bad)
        optimizer.zero_grad()
        especial_loss.backward()
        optimizer.step()


        # Check validation loss
        with torch.no_grad():
            # TODO: use a proper validation set
            validation_loss = criterion(model(x_val), y_val)
            if validation_loss < 0.3:
                break

        print(f"Iteration: {i} | Training Loss: {training_loss:.4f} | Validation Loss: {validation_loss:.4f} ")

    torch.save(model.state_dict(), "final-model.pt")


if __name__ == "__main__":
    main()

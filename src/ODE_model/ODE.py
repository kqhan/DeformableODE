import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt


# === Positional Encoding ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T, :].to(x.device)


# === Transformer Encoder ===
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: [N, K, T, F]
        N, K, T, F = x.shape
        x = x.view(N * K, T, F)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        encoded = self.transformer(x)
        return encoded[:, 0, :].view(N, K, -1)  # [N, K, D]


# === ODE Function ===
class ODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # self.aggregate = nn.Linear(hidden_dim, hidden_dim)  # gnn with full connectivity
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, hidden_dim),
        )

    def forward(self, t, x):
        return self.net(x)


# === ODE Decoder (map latent to observed) ===
class ODEDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.Tanh(),
            nn.Linear(128, output_dim)
        )

    def forward(self, z):
        # z: [N, K, T_out, D]
        N, K, T, D = z.shape
        z_flat = z.reshape(-1, D)
        out = self.net(z_flat)
        return out.view(N, K, T, -1)  # [N, K, T, F]


# === Full Neural ODE Model ===
class NeuralODEModel(nn.Module):
    def __init__(self, input_dim=2, model_dim=64, num_heads=4, num_layers=2, output_dim=2):
        super().__init__()
        self.encoder = TransformerEncoder(input_dim, model_dim, num_heads, num_layers)
        self.odefunc = ODEFunc(model_dim)
        self.decoder = ODEDecoder(model_dim, output_dim)

    def forward(self, x, t):
        z0 = self.encoder(x)  # [N, K, D]
        z_t = odeint(self.odefunc, z0, t, method='dopri5')  # [T_out, N, K, D]
        z_t = z_t.permute(1, 2, 0, 3)  # [N, K, T_out, D]
        return self.decoder(z_t)  # [N, K, T_out, F]


# === Synthetic Dataset for [N, K, T, F] ===
class SyntheticDynamicsDataset(Dataset):
    def __init__(self, N=1000, K=5, T=50, F=2):
        super().__init__()
        self.N, self.K, self.T, self.F = N, K, T, F
        self.data = self.generate_data()

    def generate_data(self):
        t = torch.linspace(0, 2 * np.pi, self.T)
        freq = torch.rand(self.N, self.K, 1, 1) * 3 + 1
        phase = torch.rand(self.N, self.K, 1, 1) * 2 * np.pi
        traj = torch.sin(freq * t.view(1, 1, self.T, 1) + phase)
        traj = traj.repeat(1, 1, 1, self.F)  # repeat over feature dim
        return traj  # [N, K, T, F]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.data[idx]  # [K, T, F]


# === Training Loop ===
# def train(model, dataloader, optimizer, criterion, device, epochs=10, t_steps=50):



# === Evaluation Loop ===
@torch.no_grad()
def evaluate(model, dataloader, criterion, device, t_steps=50, plot_examples=3):
    t = torch.linspace(0, 1, t_steps).to(device)
    model.eval()
    total_loss = 0
    for batch in dataloader:
        batch = batch.to(device)
        pred = model(batch, t)
        loss = criterion(pred, batch)
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"[Eval] MSE: {avg_loss:.6f}")
    return avg_loss

    # Calculate MSE for each timestep

def evaluate_t(model, dataloader, criterion, device, t_steps=50, plot_examples=3):
    t = torch.linspace(0, 1, t_steps).to(device)
    model.eval()
    mse_per_timestep = []
    for batch in dataloader:
        batch = batch.to(device)
        pred = model(batch[:,:,:10,:], t)
        # Calculate MSE across batch, objects and features for each timestep
        mse = ((pred - batch) ** 2).mean(dim=(0,1,3)).detach().cpu().numpy()  # Average over N,K,F dims
        mse_per_timestep.append(mse)
    
    # Average across batches
    final_mse = np.mean(np.stack(mse_per_timestep, axis=0), axis=0)  # [T]
    np.save('mse_results.npy', final_mse)


    # Plot
    # if plot_examples > 0:
    #     batch = next(iter(dataloader)).to(device)
    #     pred = model(batch, t).cpu()
    #     true = batch.cpu()
    #     for i in range(min(plot_examples, pred.shape[0])):
    #         plt.figure(figsize=(12, 3))
    #         for k in range(min(3, pred.shape[1])):  # up to 3 objects
    #             for f in range(min(2, pred.shape[-1])):  # up to 2 features
    #                 plt.plot(true[i, k, :, f], linestyle='--', label=f"True Obj{k} F{f}")
    #                 plt.plot(pred[i, k, :, f], label=f"Pred Obj{k} F{f}")
    #         plt.legend()
    #         plt.title(f"Trajectory {i}")
    #         plt.show()

    return 0


# === Main ===
def main():
    # Settings
    BATCH_SIZE = 4
    N, K, T, F = 500, 3, 20, 3
    DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # Load the saved positions data
    data_path = 'combined_positions.npy'
    all_positions = np.load(data_path)
    all_positions=all_positions[:,:,:,:]
    # Calculate split indices (80% train, 20% test)
    total_samples = len(all_positions)
    
    train_size = int(0.8 * total_samples)
    
    # Split the data
    train_data = all_positions[:train_size]
    test_data = all_positions[train_size:]
    
    # Convert to PyTorch tensors
    train_data = torch.from_numpy(train_data).float()
    test_data = torch.from_numpy(test_data).float()
    
    N = len(train_data)  # Update N to match actual training data size
    K=train_data.shape[1]
    T=train_data.shape[2]
    F=train_data.shape[3]
    # dataset = SyntheticDynamicsDataset(N=N, K=K, T=T, F=F)
    # print(dataset.shape)
    # ss
    dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    # test_dataset = SyntheticDynamicsDataset(N=100, K=K, T=T, F=F)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    model = NeuralODEModel(input_dim=F, model_dim=64, output_dim=F).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    t_steps=T
    device = DEVICE
    epochs = 100
    t = torch.linspace(0, 1, t_steps).to(device)
    # train(model, dataloader, optimizer, criterion, DEVICE, epochs=20, t_steps=T)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)  # [B, K, T, F]
            optimizer.zero_grad()
            pred = model(batch[:,:,:10,:], t)  # [B, K, T, F]
            loss = criterion(pred, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {total_loss / len(dataloader):.6f}")
        if epoch%1 == 0:
            evaluate(model, test_loader, criterion, DEVICE, t_steps=T, plot_examples=3)
        if epoch==20: 
            evaluate_t(model, test_loader, criterion, DEVICE, t_steps=T, plot_examples=3)
    evaluate(model, test_loader, criterion, DEVICE, t_steps=T, plot_examples=3)


if __name__ == "__main__":
    main()

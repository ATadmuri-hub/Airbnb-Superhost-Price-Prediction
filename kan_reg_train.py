import numpy as np
import pandas as pd
import torch
import kan

# -----------------------------
# 1) Load exported matrices (regression)
# -----------------------------
x_train = pd.read_csv("x_kan_reg_train.csv").to_numpy(dtype=np.float32)
y_train = pd.read_csv("y_kan_reg_train.csv")["y"].to_numpy(dtype=np.float32).reshape(-1, 1)
x_test  = pd.read_csv("x_kan_reg_test.csv").to_numpy(dtype=np.float32)

print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape:  {x_test.shape}")

if x_train.shape[0] != y_train.shape[0]:
    raise ValueError("x_train and y_train row counts do not match.")

# -----------------------------
# 2) Subsample for runtime control
# -----------------------------
rng = np.random.default_rng(42)
n_sub = min(5000, x_train.shape[0])
idx = rng.choice(x_train.shape[0], size=n_sub, replace=False)

x_sub = x_train[idx]
y_sub = y_train[idx]

print(f"Using subsample: {x_sub.shape[0]} rows, {x_sub.shape[1]} features")

# -----------------------------
# 3) Torch tensors
# -----------------------------
device = "cpu"
torch.manual_seed(42)
x_train_t = torch.tensor(x_sub, dtype=torch.float32, device=device)
y_train_t = torch.tensor(y_sub, dtype=torch.float32, device=device)
x_test_t  = torch.tensor(x_test, dtype=torch.float32, device=device)

# -----------------------------
# 4) Build KAN model
# -----------------------------
n_feat = x_sub.shape[1]
model = kan.KAN(
    width=[n_feat, 16, 8, 1],
    grid=5,
    k=3,
    seed=42,
    device=device
)

print("KAN regression model initialized.")

# -----------------------------
# 5) Train with standard PyTorch loop (MSE loss)
# -----------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=250)

model.train(True)
n_steps = 250

for step in range(1, n_steps + 1):
    optimizer.zero_grad()
    pred = model(x_train_t)
    loss = torch.nn.functional.mse_loss(pred, y_train_t)

    if torch.isnan(loss) or torch.isinf(loss):
        print(f"Step {step:3d} | loss is NaN/Inf — stopping early.")
        break

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    scheduler.step()

    if step == 1 or step % 50 == 0 or step == n_steps:
        print(f"Step {step:3d} | loss = {loss.item():.5f} | lr = {scheduler.get_last_lr()[0]:.6f}")

# -----------------------------
# 6) Predict on test set (log-price scale)
# -----------------------------
model.train(False)
with torch.no_grad():
    pred_test_log = model(x_test_t).cpu().numpy().reshape(-1)

# -----------------------------
# 7) Save predictions for R
# -----------------------------
out = pd.DataFrame({"kan_reg_pred_test": pred_test_log})
out.to_csv("kan_reg_pred_test.csv", index=False)

print("Saved: kan_reg_pred_test.csv")
print(out.head())


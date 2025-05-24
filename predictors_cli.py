import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import argparse
from torch_geometric.nn import GCNConv, global_max_pool as gmp, global_mean_pool as gap

import os
import requests

def download_model_from_zenodo(url, output_path="attention_model.pth"):
    if not os.path.exists(output_path):
        print(f"[↓] Downloading model from {url}...")
        r = requests.get(url)
        with open(output_path, 'wb') as f:
            f.write(r.content)
        print("[✓] Download complete.")
    else:
        print("[✓] Model already exists locally.")


# --- CBAM Modules ---

class channel_attention(nn.Module):
    def __init__(self):
        super(channel_attention, self).__init__()
        self.maxpooling = nn.AdaptiveMaxPool1d(1)
        self.avgpooling = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(32, 16, bias=False),
            nn.Linear(16, 32, bias=False),
            nn.ReLU(inplace=True)
        )
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x1 = self.maxpooling(x)
        x2 = self.avgpooling(x)
        x1_mlp = self.mlp(x1.squeeze(-1))
        x2_mlp = self.mlp(x2.squeeze(-1))
        feats = self.activation(x1_mlp + x2_mlp)
        return x * feats.unsqueeze(-1)

class SAM(nn.Module):
    def __init__(self):
        super(SAM, self).__init__()
        self.convlayer = nn.Conv1d(2, 1, kernel_size=3, padding=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat((x_mean, x_max), dim=1)
        x_conv = self.convlayer(x_cat)
        return x * self.activation(x_conv)

class CBAM(nn.Module):
    def __init__(self):
        super(CBAM, self).__init__()
        self.channel_attention = channel_attention()
        self.spatial_attention = SAM()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

# --- GCN Model ---

class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        embedding_size = 64
        linear_embeddings = 32
        self.initial_GCN = GCNConv(9, embedding_size)
        self.GCN2 = GCNConv(embedding_size, embedding_size)
        self.GCN_output_layer = nn.Linear(embedding_size * 2, 768)

        self.CNN_layers = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, padding=1),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.Conv1d(32, 32, kernel_size=3, padding=1)
        )

        self.cbam = CBAM()
        self.pooling = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()

        self.CNN_linear = nn.Sequential(
            nn.Linear(12288, linear_embeddings),
            nn.Softplus(),
            nn.Linear(linear_embeddings, linear_embeddings),
            nn.Softplus(),
            nn.Linear(linear_embeddings, 1),
            nn.Softplus()
        )
        self.dropout = nn.Dropout(0.6)

    def forward(self, drug, edge_index, batch_index, protein):
        x = F.tanh(self.initial_GCN(drug, edge_index))
        x = F.tanh(self.GCN2(x, edge_index))
        x = self.GCN2(x, edge_index)

        pooled = torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1)
        gcn_out = self.GCN_output_layer(pooled).unsqueeze(0).permute(1, 0, 2)

        combined = torch.cat((gcn_out, protein), dim=1)
        x = self.CNN_layers(combined)
        x = self.dropout(x)
        x = self.cbam(x)
        x = self.pooling(x)
        x = self.flatten(x)
        out = self.CNN_linear(x)
        return out

# --- Helper Functions (Replace these) ---

def featurize_smiles(smiles):
    raise NotImplementedError("Replace featurize_smiles with your real featurizer.")

def encode_protein(sequence):
    raise NotImplementedError("Replace encode_protein with your real protein encoder.")

# --- Prediction Function ---

def predict(csv_path, protein_seq):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(csv_path)
    smiles_list = df['SMILES'].tolist()

    model = GCN().to(device)
    zenodo_url = "https://zenodo.org/records/15502139/files/attention_model.pth?download=1"
    download_model_from_zenodo(zenodo_url)
    model.load_state_dict(torch.load("attention_model.pth", map_location=device))
    model.eval()

    predictions = []
    for smiles in smiles_list:
        drug, edge_index, batch_index = featurize_smiles(smiles)
        protein = encode_protein(protein_seq)

        drug = drug.to(device)
        edge_index = edge_index.to(device)
        batch_index = batch_index.to(device)
        protein = protein.to(device)

        with torch.no_grad():
            pred = model(drug, edge_index, batch_index, protein)
        predictions.append(pred.item())

    df['Predicted_Ki(nM)'] = predictions
    df.to_csv("predictions.csv", index=False)
    print("[✓] Predictions saved to predictions.csv")

# --- CLI Interface ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drug Affinity Predictor for D2R")
    parser.add_argument("--input", required=True, help="CSV file with 'SMILES' column")
    parser.add_argument("--protein", required=True, help="D2R protein sequence (as a string)")
    args = parser.parse_args()

    predict(args.input, args.protein)

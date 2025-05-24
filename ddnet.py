from re import I
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import argparse
from torch_geometric.nn import GCNConv, global_max_pool as gmp, global_mean_pool as gap
import rdkit
from rdkit import Chem
import numpy as np
import os
import requests

from ProtFlash.pretrain import load_prot_flash_base
from ProtFlash.utils import batchConverter

# --- Download model checkpoint ---
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class channel_attention(nn.Module):
    def __init__(self):
        super(channel_attention,self).__init__()
        self.maxpooling = nn.AdaptiveMaxPool1d(1)
        self.avgpooling = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_features = 32, out_features = 16, bias = False),
            nn.Linear(in_features = 16, out_features = 32, bias = False),
            nn.ReLU(inplace = True)
            




        )
            
        self.activation = nn.Sigmoid()


    
    def forward(self,x):
        x1 = self.maxpooling(x)
        #print(x1.shape)
        x2 = self.avgpooling(x)
        #print(x2.shape)
        x1_mlp = self.mlp(x1.squeeze(-1))
        x2_mlp = self.mlp(x2.squeeze(-1))
        feats = x1_mlp + x2_mlp
        feats = self.activation(feats)
        #print(feats.shape)
        channel_refined_feats = x * feats.unsqueeze(-1)
        return(channel_refined_feats)
        

class SAM(nn.Module):
    def __init__(self):
        super(SAM,self).__init__()

        
        self.convlayer = nn.Sequential(
            nn.Conv1d(in_channels = 2, out_channels = 1, kernel_size = 3, padding = 1),
        )
        self.activation = nn.Sigmoid()

        
    def forward(self,x):
        x_mean = torch.mean(x, dim = 1, keepdim = True)
        x_max,_ = torch.max(x,dim = 1, keepdim = True)
        x_cat = torch.cat((x_mean,x_max), dim = 1)
        x_conv = self.convlayer(x_cat)
        spatial_feats = self.activation(x_conv)
        refined_spatial_feats = x * spatial_feats
        return(refined_spatial_feats)
        
        

class CBAM(nn.Module):
    def __init__(self):
        super(CBAM,self).__init__()
        self.channel_attention = channel_attention()
        self.spatial_attention = SAM()
    def forward(self,x):
        channel_attention_layer = self.channel_attention(x)
        spatial_attention_layer = self.spatial_attention(channel_attention_layer)
        return(spatial_attention_layer)
        

# --- GCN Model ---

embedding_size = 64
cnn_hidden_layer = 64
linear_embeddings = 32
output_embeddings = 16
out_features_cnn = 16

class GCN(nn.Module):
    def __init__(self):
        super(GCN,self).__init__()

        #GCN_layers
        self.initial_GCN = GCNConv(9,embedding_size)
        self.GCN1 = GCNConv(embedding_size,embedding_size)
        self.GCN2 = GCNConv(embedding_size,embedding_size)
        self.GCN_output_layer = nn.Linear(in_features = embedding_size*2, out_features = 768)

        self.bn1 = nn.LayerNorm(embedding_size)
        self.bn2 = nn.LayerNorm(embedding_size)



        #CNN_layer
        self.CNN_layers = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size = 3, stride=1,padding=1),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size = 3, stride=1,padding=1),

            nn.Conv1d(in_channels=32, out_channels=32, kernel_size = 3, stride=1,padding=1),







            
        )
        self.pooling = nn.MaxPool1d(kernel_size = 2)
        self.flatten = nn.Flatten()
        self.cbam = CBAM()

        self.CNN_linear = nn.Sequential(
            nn.Linear(in_features= 12288, out_features = linear_embeddings),
            nn.Softplus(),







            nn.Linear(in_features= linear_embeddings, out_features = linear_embeddings),
            nn.Softplus(),






            nn.Linear(in_features= linear_embeddings,out_features = 1),
            nn.Softplus()



            )
        self.dropout = nn.Dropout(0.6)


    
        

      

     

      

    def forward(self,drug, edge_index, batch_index, protein):
        GCN_layer_1 = self.initial_GCN(drug, edge_index)
        GCN_layer_1 = F.tanh(GCN_layer_1)

        GCN_layer_2 = self.GCN2(GCN_layer_1, edge_index)
        GCN_layer_2 = F.tanh(GCN_layer_2)


        GCN_layer_3 = self.GCN2(GCN_layer_2,edge_index)


        hidden = torch.cat([gmp(GCN_layer_3,batch_index), gap(GCN_layer_3,batch_index)],dim = 1)


        GCN_output_layer = self.GCN_output_layer(hidden)
        GCN_output_layer = GCN_output_layer.unsqueeze(0)


        GCN_output_layer = GCN_output_layer.permute(1,0,2)

        combined = torch.cat((GCN_output_layer, protein), dim = 1)

        CNN_layer = self.CNN_layers(combined)
        CNN_layer = self.dropout(CNN_layer)


        attention_layer = self.cbam(x = CNN_layer)
        pooled_attention_layer = self.pooling(attention_layer)
        flattened_attention_layer = self.flatten(pooled_attention_layer)
        



        CNN_output = self.CNN_linear(flattened_attention_layer)
        
        return(CNN_output)

# --- Protein Encoding (with ProtFlash) ---


def encode_protein(sequence):
    model = load_prot_flash_base()
    model.to(device)
    model.eval()

    data = [("protein1", sequence)]
    ids, batch_token, lengths = batchConverter(data)

    batch_token = batch_token.to(device)

    with torch.no_grad():
        token_embedding = model(batch_token, lengths)
        
    # Generate per-sequence representations via averaging
    seq_embedding = token_embedding[0, 0: len(sequence) + 1].mean(0)  # [embedding_dim]
    seq_embedding = seq_embedding.unsqueeze(0).unsqueeze(0)  # shape: [1,1,1, embedding_dim]

    return seq_embedding


# --- Drug Featurization (Placeholder) ---
def get_node_features(mol):
  all_node_feats = []
  for atom in mol.GetAtoms():
    node_feats = []
    # Feature 1: Atomic number
    node_feats.append(atom.GetAtomicNum())
    # Feature 2: Atom degree
    node_feats.append(atom.GetDegree())
    # Feature 3: Formal charge
    node_feats.append(atom.GetFormalCharge())
    # Feature 4: Hybridization
    node_feats.append(atom.GetHybridization())
    # Feature 5: Aromaticity
    node_feats.append(atom.GetIsAromatic())
    # Feature 6: Total Num Hs
    node_feats.append(atom.GetTotalNumHs())
    # Feature 7: Radical Electrons
    node_feats.append(atom.GetNumRadicalElectrons())
    # Feature 8: In Ring
    node_feats.append(atom.IsInRing())
    # Feature 9: Chirality
    node_feats.append(atom.GetChiralTag())

    # Append node features to matrix
    all_node_feats.append(node_feats)

  all_node_feats = np.asarray(all_node_feats)
  return torch.tensor(all_node_feats, dtype=torch.float)

def get_edge_features(mol):
        """
        This will return a matrix / 2d array of the shape
        [Number of edges, Edge Feature size]
        """
        all_edge_feats = []

        for bond in mol.GetBonds():
            edge_feats = []
            # Feature 1: Bond type (as double)
            edge_feats.append(bond.GetBondTypeAsDouble())
            # Feature 2: Rings
            edge_feats.append(bond.IsInRing())
            # Append node features to matrix (twice, per direction)
            all_edge_feats += [edge_feats, edge_feats]

        all_edge_feats = np.asarray(all_edge_feats)
        return torch.tensor(all_edge_feats, dtype=torch.float)

def get_adjacency_info(mol):
     
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]]

        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        return edge_indices




# --- Prediction Function ---

def predict(csv_path, protein_seq, output_path="predictions.csv"):
    df = pd.read_csv(csv_path)
    smiles_list = df['SMILES'].tolist()

    cnn = GCN().to(device)
    zenodo_url = "https://zenodo.org/records/15502139/files/attention_model.pth?download=1"
    download_model_from_zenodo(zenodo_url)
    cnn.load_state_dict(torch.load("attention_model.pth", map_location=device))
    cnn.eval()
    
    predictions = []
    node_features = []
    edge_features = []
    edge_index = []
    indices_skipped = []
    indices_used = []

    for i in range(len(smiles_list)):
      mol_obj = Chem.MolFromSmiles(smiles_list[i])
      if mol_obj is not None:
        node_feature = get_node_features(mol_obj)
        node_features.append(node_feature)
        edge_feature = get_edge_features(mol_obj)
        edge_features.append(edge_feature)
        edge_adjacency = get_adjacency_info(mol_obj)
        edge_index.append(edge_adjacency)
        indices_used.append(i)
      else:
        indices_skipped.append(i)
      

      
    for i in range(len(node_features)):
      drug = node_features[i]
      edges = edge_index[i]
      protein = encode_protein(protein_seq)



      with torch.no_grad():
        pred = cnn(drug, edges, batch_index = None, protein = protein)
        predictions.append(pred.item())
        

    df['Predicted_Ki(nM)'] = np.nan
    for idx, pred in zip(indices_used, predictions):
      df.at[idx, 'Predicted_Ki(nM)'] = pred

    df.to_csv(output_path, index=False)
    print(f"[✓] Predictions saved to {output_path}")
    if len(indices_skipped) > 0:
      print(f"SMILES indices that were skipped: {indices_skipped}")

# --- CLI Interface ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drug Affinity Predictor for D2R")
    parser.add_argument("--input", required=True, help="CSV file with 'SMILES' column")
    parser.add_argument("--protein", required=True, help="D2R protein sequence (as a string)")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV file name")
    args = parser.parse_args()

    predict(args.input, args.protein, args.output)

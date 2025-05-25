# Data

This folder contains datasets related to the development and application of **DeepDopamineNet**.

## Files

### `Training_data.csv`
Contains molecular and target information used for training the DeepDopamineNet model.  
Includes curated SMILES representations and corresponding activity values (Ki) used to learn predictive relationships for the D2 dopamine receptor (D2R).

### `Chembl_data.csv`
Contains a list of SMILES IDs obtained from the ChEMBL database.  
These compounds were screened using the trained model to predict their binding affinity against D2R. Compounds with the highest predicted affinities are highlighted in the accompanying results.

---

> 📁 *This data is provided to support transparency and reproducibility of the DeepDopamineNet project. For model weights and code, please refer to the root directory or the [main repository](https://github.com/uzairamu/DeepDopamineNet).*

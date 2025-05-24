# D2R-KiPredictor

This project provides a tool to predict binding affinity (Ki in nM) of small molecules (in SMILES format) against the dopamine D2 receptor (D2R).

## Files
- `predictor_cli.py`: Command-line script for batch prediction.
- `requirements.txt`: Dependencies needed to run the script.
- `example_input.csv`: Sample input file with SMILES.

## Usage

### Input Format
CSV file with a `smiles` column:

```
smiles
CC(=O)OC1=CC=CC=C1C(=O)O
CN1CCC(CC1)CN2C(=O)C=CC2=O
```

### Run Prediction
```bash
python predictor_cli.py --input example_input.csv --output predictions.csv
```

### Output
CSV file with predicted Ki values:

```
SMILES,Predicted_Ki_nM
CC(=O)OC1=CC=CC=C1C(=O)O,2.5
CN1CCC(CC1)CN2C(=O)C=CC2=O,1.7
```

## License
MIT License

## Citation
If you use this model, please cite it using the Zenodo DOI.

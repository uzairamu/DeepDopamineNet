# D2R-Ki Predictor

This project provides a tool to predict binding affinity (Ki in nM) of small molecules (in SMILES format) against the dopamine D2 receptor (D2R).

## Files
- `ddnet.py`: Command-line script for batch prediction.
- `requirements.txt`: Dependencies needed to run the script.
- `example_input.csv`: Sample input file with SMILES.


## Clone the Repository

To get started, first clone this repository:

```bash
git clone https://github.com/uzairamu/DeepDopamineNet.git
cd DeepDopamineNet

## Install Dependencies

Create a Python environment (recommended) and install the required packages:
python -m venv ddnet
source ddnet/bin/activate  
pip install -r requirements.txt

```



### Input Format
CSV file with a `smiles` column:

```
SMILES (#case sensitive)
CC(=O)OC1=CC=CC=C1C(=O)O
CN1CCC(CC1)CN2C(=O)C=CC2=O
```

### Run Prediction
```bash
python ddnet.py --input example_input.csv --output predictions.csv
```

### Output
CSV file with predicted Ki values:

```
SMILES,Predicted_Ki_nM
CC(=O)OC1=CC=CC=C1C(=O)O,2.5
CN1CCC(CC1)CN2C(=O)C=CC2=O,1.7
```

### 🖥️ Graphical User Interface (GUI) – Beginner-Friendly
Prefer a simple point-and-click interface? We’ve got you covered! A standalone GUI version is available for users who prefer not to use the terminal.
(Note currently we only have GUI version for Linux users)

To use the GUI, download the file "deepdopaminenet.tar.xz" from this repo; or run

```
wget https://github.com/uzairamu/DeepDopamineNet/raw/main/deepdopaminenet.tar.xz

```

Extract the Files

```
tar -xf deepdopaminenet.tar.xz
cd deepdopaminenet

```

Install dependencies

```

sudo apt-get install python3-tk 
pip install pydantic==2.10.6
pip install gradio

```


Run the installer script

```
chmod +x installer.sh
./installer.sh

```
This script will:

Set up the application environment

Install all necessary dependencies

Add a desktop shortcut to your Applications menu

Launch the App

After installation, you can launch DeepDopamineNet from your Applications menu like any other app.


## License
MIT License

## Citation
If you use this model, please cite it using our Zenodo doi : https://doi.org/10.5281/zenodo.15502139


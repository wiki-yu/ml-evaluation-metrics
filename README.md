# ml-evaluation-metrics

## SET UP
### CLONE THE PROJECT
```
# clone the project
git clone https://github.com/wiki-yu/ml-evaluation-metrics.git
```

## Build virtual enviroment:

_For Windows:_

```bash
py -m venv env # Only run this if env folder does not exist
.\env\Scripts\activate
pip install -r requirements.txt
```

_For MacOS/Linux:_

```bash
python3 -m venv env # Only run this if env folder does not exist
source env/bin/activate

pip install -r requirements.txt
``````

## Run the script
```bash
// enter the sub folder
cd binary-classification-metrics
// run the metrics script
python main.py input.json output.json
```

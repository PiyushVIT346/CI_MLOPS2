D:\projects\DVC project2>cd dvc+mlflow+ci2
D:\projects\DVC project2\dvc+mlflow+ci2>cookiecutter -c v1 https://github.com/drivendata/cookiecutter-data-science
D:\projects\DVC project2\dvc+mlflow+ci2>cd CI_MLOPS 

D:\projects\DVC project2\dvc+mlflow+ci2\CI_MLOPS>python -m
 venv myenv
D:\projects\DVC project2\dvc+mlflow+ci2\CI_MLOPS>myenv\Scripts\activate
(myenv) D:\projects\DVC project2\dvc+mlflow+ci2\CI_MLOPS>g
it init
(myenv) D:\projects\DVC project2\dvc+mlflow+ci2\CI_MLOPS>git status
(myenv) D:\projects\DVC project2\dvc+mlflow+ci2\CI_MLOPS>g
it add .
(myenv) D:\projects\DVC project2\dvc+mlflow+ci2\CI_MLOPS>git commit -m "Add cookiecutter"
(myenv) D:\projects\DVC project2\dvc+mlflow+ci2\CI_MLOPS>g
it remote add origin https://github.com/PiyushVIT346/CI_MLOPS2.git
(myenv) D:\projects\DVC project2\dvc+mlflow+ci2\CI_MLOPS>g
it push origin master

connect to dagshub
go to the dagshub website. click on create button on top right corner. now create a new repo and connect github.
now on dagshub click on remote option and than on experimwnt tab. now copy the code given and save on notepad. also click on mlflow ui button just above the given code in experimwnt tab. 

in terminal,
pip install mlflow
pip install dagshub
pip install seaborn

go to  notebook folder and create a new file "dagshub_test.py" and paste the code copied and pasted on notepad.
Now run code using command: python D:\projects\DVC project2\dvc+mlflow+ci2\CI_MLOPS\notebooks\dagshub_test.py
open mlflow and check if log added.

to start dvc
write the command: dvc init
to make storage for versioning enter the command: dvc remote add -d myremote C:\Users\HP\AppData\Local\Temp

now add the code of data_collection.py, data_prep.py, model_building.py, model_eval.py, model_reg.py , dvc.yaml and params.yaml.
now run code using command: dvc repro




CI_MLOPS
==============================

MLOPS

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

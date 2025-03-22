import dagshub
dagshub.init(repo_owner='PiyushVIT346', repo_name='CI_MLOPS2', mlflow=True)

import mlflow
with mlflow.start_run():
    mlflow.log_param('parameter name', 'value')
    mlflow.log_metric('metric name', 1)
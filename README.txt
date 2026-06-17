1. create python environment following steps in uv-environment.md
2. Signup for your free databricks account from https://www.databricks.com
2.1. Explore the free edition of the product: https://www.databricks.com/learn/free-edition
3. create .env file in the root directory with the following values:
    DATABRICKS_TOKEN=""
    DATABRICKS_HOST="https://<your-workspace-id>.databricks.com"
    DATABRICKS_MODEL="databricks-qwen35-122b-a10b"
    # Required for wk2-databricks notebooks (Databricks Vector Search):
    DATABRICKS_VS_ENDPOINT="<your-vector-search-endpoint-name>"
    DATABRICKS_VS_INDEX="cs4603.rag.docs_index"
4. Generate DATABRICKS_TOKEN from developer settings under your databricks workspace.
5. You can use any of the models as retreived in the first notebook (0.endpoints.ipynb)
5.1 Model list is also available from left menu of your workspace -- AI/ML --> Models
6. For the notebooks to demonstrate tracing and mlflow, you can start mlflow server with this command:
    mlflow ui --port 5000
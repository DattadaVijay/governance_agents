# Databricks notebook source

# COMMAND ----------

import mlflow
import mlflow.pyfunc
import pandas as pd
import os
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

# COMMAND ----------

# ── Config ────────────────────────────────────────────────────────
CATALOG    = "governance"
SCHEMA     = "default"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.data_governance_agent"
AGENT_PATH = "./agent.py"

# ── Set GROQ key so load_context works during local test ──────────
os.environ["GROQ_API_KEY"] = dbutils.secrets.get("agents_scope", "groq_api_key")

# COMMAND ----------

# ── MLflow experiment ─────────────────────────────────────────────
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(
    "/Users/dattada.vijay@gmail.com/data_governance_agent"
)

# COMMAND ----------

# ── Signature ─────────────────────────────────────────────────────
signature = ModelSignature(
    inputs=Schema([
        ColSpec("string", "question"),
        ColSpec("string", "thread_id")
    ]),
    outputs=Schema([
        ColSpec("string")
    ])
)

# ── Input example ─────────────────────────────────────────────────
input_example = pd.DataFrame({
    "question":  ["What is the status of job 780838995876631?"],
    "thread_id": ["session_001"]
})

# COMMAND ----------

# ── Log model ─────────────────────────────────────────────────────
with mlflow.start_run(run_name="data_governance_agent_v1") as run:

    mlflow.pyfunc.log_model(
        artifact_path="data_governance_agent",
        python_model=AGENT_PATH,
        input_example=input_example,
        signature=signature,
        pip_requirements=[
            "langchain",
            "langchain-groq",
            "langgraph",
        ]
    )

    run_id = run.info.run_id
    print(f"✅ Model logged")
    print(f"   Run ID:    {run_id}")
    print(f"   Model URI: runs:/{run_id}/data_governance_agent")

# COMMAND ----------

# ── Register to Unity Catalog ─────────────────────────────────────
registered = mlflow.register_model(
    model_uri=f"runs:/{run_id}/data_governance_agent",
    name=MODEL_NAME
)

print(f"✅ Registered: {MODEL_NAME}")
print(f"   Version:    {registered.version}")

# COMMAND ----------

# ── Set alias ─────────────────────────────────────────────────────
client = mlflow.tracking.MlflowClient()

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=registered.version
)

print(f"✅ Alias 'champion' → v{registered.version}")
print(f"   Load URI: models:/{MODEL_NAME}@champion")

# COMMAND ----------

# ── Quick load test ───────────────────────────────────────────────
print("\nTesting loaded model...")

model = mlflow.pyfunc.load_model(
    f"models:/{MODEL_NAME}@champion"
)

test_df = pd.DataFrame({
    "question":  ["What jobs failed in the last 210 hours?"],
    "thread_id": ["test_session"]
})

result = model.predict(test_df)
print(f"\nQ: {test_df['question'][0]}")
print(f"A: {result[0]}")
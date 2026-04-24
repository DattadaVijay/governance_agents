# Databricks notebook source

# MAGIC %pip install \
# MAGIC     "langchain==0.3.7" \
# MAGIC     "langchain-core==0.3.15" \
# MAGIC     "langchain-groq==0.2.1" \
# MAGIC     "langgraph==0.2.45"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import mlflow.pyfunc
import pandas as pd
import os
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

# COMMAND ----------

CATALOG    = "governance"
SCHEMA     = "default"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.data_governance_agent"
AGENT_PATH = "./agent.py"

os.environ["GROQ_API_KEY"] = dbutils.secrets.get("agents_scope", "grok_key")

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(
    "/Users/dattada.vijay@gmail.com/data_governance_agent"
)

# COMMAND ----------

signature = ModelSignature(
    inputs=Schema([
        ColSpec("string", "question"),
        ColSpec("string", "thread_id")
    ]),
    outputs=Schema([
        ColSpec("string")
    ])
)

input_example = pd.DataFrame({
    "question":  ["What is the status of job 780838995876631?"],
    "thread_id": ["session_001"]
})

# COMMAND ----------

with mlflow.start_run(run_name="data_governance_agent_v1") as run:

    mlflow.pyfunc.log_model(
        artifact_path="data_governance_agent",
        python_model=AGENT_PATH,
        input_example=input_example,
        signature=signature,
        pip_requirements=[
            "langchain==0.3.7",
            "langchain-core==0.3.15",
            "langchain-groq==0.2.1",
            "langgraph==0.2.45",
        ]
    )

    run_id = run.info.run_id
    print(f"✅ Model logged | Run ID: {run_id}")

# COMMAND ----------

registered = mlflow.register_model(
    model_uri=f"runs:/{run_id}/data_governance_agent",
    name=MODEL_NAME
)

print(f"✅ Registered: {MODEL_NAME} v{registered.version}")


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
#
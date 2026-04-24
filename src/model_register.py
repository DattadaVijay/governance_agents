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
import shutil
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

# COMMAND ----------

# ── Config ────────────────────────────────────────────────────────
CATALOG    = "governance"
SCHEMA     = "default"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.data_governance_agent"

# ── Copy agent to /tmp so path is always known ────────────────────
shutil.copy(
    "/Workspace/Users/dattada.vijay@gmail.com/.bundle/databricks_agent/dev/files/src/agent.py",
    "/tmp/agent.py"
)
AGENT_PATH = "/tmp/agent.py"
print(f"✅ Agent copied to: {AGENT_PATH}")

# ── Set secret ────────────────────────────────────────────────────
os.environ["GROQ_API_KEY"] = dbutils.secrets.get("agents_scope", "grok_key")
print("✅ GROQ_API_KEY set")

# COMMAND ----------

# ── MLflow setup ──────────────────────────────────────────────────
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(
    "/Users/dattada.vijay@gmail.com/data_governance_agent"
)
print("✅ MLflow experiment set")

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
            "langchain==0.3.7",
            "langchain-core==0.3.15",
            "langchain-groq==0.2.1",
            "langgraph==0.2.45",
        ]
    )

    run_id = run.info.run_id
    print(f"✅ Model logged")
    print(f"   Run ID: {run_id}")

# COMMAND ----------

# ── Register ──────────────────────────────────────────────────────
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

# COMMAND ----------

# ── Load test ─────────────────────────────────────────────────────
print("\nLoading model for test...")

model = mlflow.pyfunc.load_model(
    f"models:/{MODEL_NAME}@champion"
)

print("✅ Model loaded")

# ── Test questions ────────────────────────────────────────────────
questions = [
    ("What is the job ID of dev_dattada_vijayyml_job_deploy?",   "session_001"),
    ("Who created job 780838995876631?",                          "session_001"),
]

for question, thread_id in questions:
    test_df = pd.DataFrame({
        "question":  [question],
        "thread_id": [thread_id]
    })
    result = model.predict(test_df)
    print(f"\nQ: {question}")
    print(f"A: {result[0]}")
    print("-" * 60)
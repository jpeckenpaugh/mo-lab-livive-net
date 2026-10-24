"""ModelOps Lab package.

W2 provides the deterministic dataset pipeline (generation, validation,
canonical serialization, versioned artifacts) and reproducible training with
MLflow experiment tracking. Registry, serving, monitoring, and retraining arrive
in later waves.
"""

import os

# MLflow prints an agent-oriented hint on import. It is noise for CLI users and
# demo output; the environment variable is MLflow's supported off switch.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

__version__ = "0.1.0"

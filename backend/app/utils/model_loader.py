import os, pickle

MODEL_DIR = os.getenv("MODEL_DIR", "models")

class ModelManager:
    _cache = {}

    @classmethod
    def load_role(cls, role: str):
        """Load ML artifacts for a specific role (cached)."""
        role = role.upper()
        if role in cls._cache:
            return cls._cache[role]

        path = os.path.join(MODEL_DIR, role)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model folder not found for role: {role}")

        artifacts = ["encoder.pkl", "features.pkl", "gmm_model.pkl", "scaler.pkl", "umap_reducer.pkl"]
        models = {}
        for file in artifacts:
            full_path = os.path.join(path, file)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Missing artifact: {file}")
            with open(full_path, "rb") as f:
                models[file.split(".")[0]] = pickle.load(f)

        cls._cache[role] = models
        print(f"Loaded model for {role}")
        return models

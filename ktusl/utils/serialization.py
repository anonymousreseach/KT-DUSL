import yaml, json

def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_yaml(obj, path: str):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f)

def to_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

from pathlib import Path
from src.pricing_engine import run

if __name__ == "__main__":
    root = Path(__file__).parent
    metrics = run(root / "data", root / "outputs" / "recommendations")
    print(metrics)

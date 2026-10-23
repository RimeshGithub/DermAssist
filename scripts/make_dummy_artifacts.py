"""Create RANDOM-WEIGHT stand-ins for the notebook artifacts, to test the app plumbing only.

    python scripts/make_dummy_artifacts.py --out models/_dummy
    MODEL_PATH=models/_dummy/model.pt ENCODERS_PATH=models/_dummy/encoders.pkl python app.py

The predictions from these files are meaningless. Never deploy them.
Writes the same keys as notebook Section 13 so the real loading code path is exercised.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import timm
import torch
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config                      # noqa: E402
from src.model_def import MultimodalClassifier  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/_dummy")
    ap.add_argument("--backbone", default="convnext_small.fb_in22k_ft_in1k",
                    help="use convnext_atto for a fast, tiny test model")
    ap.add_argument("--with-unknown", action="store_true", help="9 classes (INCLUDE_UNKNOWN=True)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    class_names = ["nv", "mel", "bcc", "bkl", "akiec", "scc", "df", "vasc"]
    if args.with_unknown:
        class_names.append("unknown")

    # Same encoder type/settings as the notebook: OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    sex_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
        np.array(config.SEX_VALUES).reshape(-1, 1))
    site_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
        np.array(config.SITE_VALUES).reshape(-1, 1))
    metadata_dim = 1 + len(sex_encoder.categories_[0]) + len(site_encoder.categories_[0])

    tmp = timm.create_model(args.backbone, pretrained=False, num_classes=0)
    cfg = timm.data.resolve_data_config({}, model=tmp)
    model = MultimodalClassifier(len(class_names), metadata_dim, 64, args.backbone, pretrained=False,
                                 dropout=0.4, drop_path_rate=0.2).eval()

    torch.save({
        "model_state_dict": model.state_dict(), "temperature": 1.37, "class_names": class_names,
        "metadata_dim": metadata_dim, "backbone": args.backbone, "img_size": cfg["input_size"][1],
        "mean": list(cfg["mean"]), "std": list(cfg["std"]),
    }, out / "model.pt")
    with open(out / "encoders.pkl", "wb") as fh:
        pickle.dump({"sex_encoder": sex_encoder, "site_encoder": site_encoder,
                     "age_median": 50.0, "age_max": 85.0}, fh)
    print(f"Wrote {out / 'model.pt'} and {out / 'encoders.pkl'} (classes={len(class_names)}, "
          f"metadata_dim={metadata_dim}, img_size={cfg['input_size'][1]})")


if __name__ == "__main__":
    main()

# models/

Copy these two files from the notebook's output (Section 13) into this folder:

| File | Produced by |
|---|---|
| `dermassist_multimodal_convnext_small_final.pt` | `torch.save({...}, f"dermassist_multimodal_{MODEL_TAG}_final.pt")` |
| `metadata_encoders.pkl` | `pickle.dump({...}, open("metadata_encoders.pkl", "wb"))` |

The `.pt` file (~200 MB) must be tracked with Git LFS (see `.gitattributes`).

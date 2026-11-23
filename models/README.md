# Place your model files here

Copy the two files from your screenshot into this folder, with these
exact names (or update the paths in `.env` to match whatever names
you actually use):

```
models/
├── model_complete.pth           <- your T2-trained ModernBERT checkpoint
└── best_4class_classifier.pt    <- your fine-tuned T3 4-class classifier
```

Do NOT put `checkpoint_epoch_2.pt` / `_4.pt` / `_6.pt` / `_8.pt` or
`stage6_classifier_head.pth` here — the application will not look
for or use them.

Before running the backend for the first time, run the inspector so
you can see exactly what's inside these two files (checkpoint type,
state_dict keys, architecture metadata, label mapping) instead of
trusting an assumption:

```bash
cd backend
python inspect_checkpoints.py
```

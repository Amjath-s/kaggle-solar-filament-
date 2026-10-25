# Train / validation splits

Created by:

```bash
python scripts/create_split.py --val-fraction 0.2 --seed 42
```

- `train.txt` — image file names for training  
- `val.txt` — image file names for local validation  

Built from **unique** JPEGs under `train/train_images` only.  
Kaggle **test** images are not included.

from pathlib import Path

scenes_f = Path('data/GL3D/scenes.txt')
val_f = Path('data/dataset_split/val.txt')
train_f = Path('data/dataset_split/train.txt')

if not scenes_f.exists():
    raise SystemExit(f"Missing {scenes_f}")
if not val_f.exists():
    raise SystemExit(f"Missing {val_f}")

scenes = [l.strip() for l in scenes_f.read_text(encoding='utf-8').splitlines() if l.strip()]
val = set(l.strip() for l in val_f.read_text(encoding='utf-8').splitlines() if l.strip())

train = [s for s in scenes if s not in val]
train_f.parent.mkdir(parents=True, exist_ok=True)
train_f.write_text('\n'.join(train) + ('\n' if train else ''), encoding='utf-8')
print(f"Wrote {len(train)} entries to {train_f}")

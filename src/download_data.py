"""Download only original fruit images from the cited public dataset."""
import csv
from pathlib import Path
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]

def main():
    out = ROOT/'data'/'raw'; out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset('Project-AgML/fresh_rotten_fruit_classification', 'raw', split='train', streaming=True)
    rows = []; failures = 0
    for i, row in enumerate(ds):
        try:
            label = ['fresh','rotten'][int(row['label'])]
            fruit = str(row['fruit_type'])
            file = out/f'{i:05d}.jpg'
            im = row['image'].convert('RGB'); im.thumbnail((512,512)); im.save(file, quality=92)
            rows.append([file.name,label,fruit])
        except (OSError,ValueError,TypeError,KeyError,IndexError) as e:
            failures += 1; print('Skipped:',i,e)
        if (i+1)%100 == 0: print('Processed',i+1,flush=True)
    if len(rows) < 100: raise RuntimeError('Incomplete download; check connection.')
    with (out/'manifest.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['file','label','fruit']); w.writerows(rows)
    print('Saved',len(rows),'images; skipped',failures)

if __name__=='__main__': main()

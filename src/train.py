"""Clean, split, augment, train, evaluate and save a binary CNN."""
import argparse, csv, hashlib, json, random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.model import FruitCNN, LABELS, transform, SIZE
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'/'task2'

class Fruits(Dataset):
    def __init__(self,rows,root,augment=False): self.rows=rows; self.root=root; self.augment=augment
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]
        with Image.open(self.root/row['file']) as im: x=transform(im,self.augment)
        return x,LABELS.index(row['label'])

def clean(root):
    manifest=root/'manifest.csv'
    if not manifest.exists(): raise FileNotFoundError('Run python -m src.download_data first.')
    with manifest.open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
    valid=[]; seen=set(); bad=0; duplicate=0
    for row in rows:
        try:
            if row['label'] not in LABELS: raise ValueError('Invalid label')
            with Image.open(root/row['file']) as im:
                im.load(); rgb=im.convert('RGB')
                digest=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
            if digest in seen: duplicate+=1; continue
            seen.add(digest); valid.append(row)
        except (OSError,ValueError,KeyError): bad+=1
    return valid,{'manifest_rows':len(rows),'valid':len(valid),'invalid':bad,'exact_duplicates':duplicate}

def evaluate(model,loader,device,loss_fn):
    model.eval(); total=0.; y=[]; pred=[]
    with torch.inference_mode():
        for x,t in loader:
            x,t=x.to(device),t.to(device); logits=model(x); total+=loss_fn(logits,t).item()*len(t)
            y.extend(t.cpu().tolist()); pred.extend(logits.argmax(1).cpu().tolist())
    return total/len(y),accuracy_score(y,pred),y,pred

def main():
    p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=12); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--lr',type=float,default=.001); p.add_argument('--seed',type=int,default=42); p.add_argument('--data',type=Path,default=ROOT/'data'/'raw'); a=p.parse_args()
    if a.epochs<1 or a.batch_size<1 or a.lr<=0: raise ValueError('Invalid training settings')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    OUT.mkdir(parents=True,exist_ok=True); (ROOT/'models').mkdir(exist_ok=True)
    rows,cleaning=clean(a.data)
    strata=[r['label']+'_'+r['fruit'] for r in rows]
    train,tmp=train_test_split(rows,test_size=.30,random_state=a.seed,stratify=strata)
    val,test=train_test_split(tmp,test_size=.5,random_state=a.seed,stratify=[r['label']+'_'+r['fruit'] for r in tmp])
    splits={'train':train,'validation':val,'test':test}
    with (OUT/'splits.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['file','label','fruit','split']); w.writeheader()
        for name,rs in splits.items(): w.writerows([{**r,'split':name} for r in rs])
    loaders={k:DataLoader(Fruits(rs,a.data,k=='train'),batch_size=a.batch_size,shuffle=k=='train',num_workers=0) for k,rs in splits.items()}
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=FruitCNN().to(device)
    loss_fn=nn.CrossEntropyLoss(); opt=torch.optim.Adam(model.parameters(),lr=a.lr)
    history=[]; best=-1.; path=ROOT/'models'/'fruit_cnn.pt'
    for epoch in range(1,a.epochs+1):
        model.train(); total=0.; correct=0; count=0
        for x,t in loaders['train']:
            x,t=x.to(device),t.to(device); opt.zero_grad(); logits=model(x); loss=loss_fn(logits,t); loss.backward(); opt.step()
            total+=loss.item()*len(t); correct+=(logits.argmax(1)==t).sum().item(); count+=len(t)
        vl,va,_,_=evaluate(model,loaders['validation'],device,loss_fn)
        history.append({'epoch':epoch,'train_loss':total/count,'train_accuracy':correct/count,'val_loss':vl,'val_accuracy':va})
        print(history[-1],flush=True)
        if va>best:
            best=va; torch.save({'state_dict':model.state_dict(),'classes':LABELS,'size':SIZE,'epoch':epoch},path)
    checkpoint=torch.load(path,map_location=device,weights_only=True); model.load_state_dict(checkpoint['state_dict'])
    tl,ta,y,pred=evaluate(model,loaders['test'],device,loss_fn); cm=confusion_matrix(y,pred,labels=[0,1])
    counts={k:{l:sum(r['label']==l for r in rs) for l in LABELS} for k,rs in splits.items()}
    result={'source':'Project-AgML/fresh_rotten_fruit_classification (raw)', 'cleaning':cleaning,'split_counts':counts,'epochs':a.epochs,'seed':a.seed,'batch_size':a.batch_size,'learning_rate':a.lr,'device':str(device),'best_epoch':checkpoint['epoch'],'test_loss':tl,'test_accuracy':ta,'test_macro_f1':f1_score(y,pred,average='macro',zero_division=0),'confusion_matrix':cm.tolist(),'classification_report':classification_report(y,pred,labels=[0,1],target_names=LABELS,output_dict=True,zero_division=0),'history':history}
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2)); print('Test accuracy:',ta,'macro F1:',result['test_macro_f1'])
    with (OUT/'history.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=history[0]);w.writeheader();w.writerows(history)
    for metric in ['loss','accuracy']:
        fig,ax=plt.subplots(figsize=(6,3.5)); ax.plot([h['epoch'] for h in history],[h['train_'+metric] for h in history],label='Train'); ax.plot([h['epoch'] for h in history],[h['val_'+metric] for h in history],label='Validation'); ax.set(xlabel='Epoch',ylabel=metric.title());ax.legend();fig.tight_layout();fig.savefig(OUT/f'{metric}.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(4,4)); im=ax.imshow(cm,cmap='Blues');ax.set(xticks=[0,1],yticks=[0,1],xticklabels=LABELS,yticklabels=LABELS,xlabel='Predicted',ylabel='Actual');
    for i in range(2):
        for j in range(2): ax.text(j,i,cm[i,j],ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    fig.tight_layout();fig.savefig(OUT/'confusion_matrix.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,3.5)); names=list(counts); x=np.arange(len(names));ax.bar(x-.18,[counts[k]['fresh'] for k in names],.36,label='Fresh');ax.bar(x+.18,[counts[k]['rotten'] for k in names],.36,label='Rotten');ax.set_xticks(x,names);ax.set_ylabel('Images');ax.legend();fig.tight_layout();fig.savefig(OUT/'class_distribution.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(9,6));
    for ax,row in zip(axes.flat,test[:6]):
        with Image.open(a.data/row['file']) as im: ax.imshow(im); ax.set_title(row['fruit']+' / '+row['label']);ax.axis('off')
    fig.tight_layout();fig.savefig(OUT/'dataset_examples.jpg',dpi=120);plt.close(fig)

if __name__=='__main__': main()

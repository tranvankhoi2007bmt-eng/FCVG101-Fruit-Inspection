"""Reproducible classical CV experiment. Synthetic fixtures are labelled as such."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.vision import read_image, filters, edges, frequency, features, match_views, calibrate, undistort, pinhole

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs'/'task1'


def synthetic_fruit():
    rng = np.random.default_rng(42)
    img = np.full((360, 480, 3), (222, 231, 236), np.uint8)
    cv2.ellipse(img, (235, 205), (124, 112), 0, 0, 360, (29, 44, 183), -1)
    cv2.ellipse(img, (190, 159), (47, 40), -30, 0, 360, (62, 97, 245), -1)
    cv2.line(img, (237, 100), (251, 51), (34, 69, 79), 13)
    cv2.ellipse(img, (277, 71), (41, 15), -25, 0, 360, (38, 125, 57), -1)
    for _ in range(120):
        x, y = rng.integers([115, 100], [355, 310])
        if ((x-235)/124)**2+((y-205)/112)**2 < .9:
            cv2.circle(img, (int(x), int(y)), int(rng.integers(1, 4)), (25, 62, 135), -1)
    cv2.putText(img, 'SYNTHETIC TEST IMAGE', (12, 342), cv2.FONT_HERSHEY_SIMPLEX, .65, (40,40,40), 2)
    return img


def checkerboards():
    # Render a planar checkerboard at varied camera poses, then apply known distortion.
    w, h = 800, 600
    K = np.array([[780., 0, 400], [0, 770., 300], [0, 0, 1.]])
    dist = np.array([-0.12, 0.035, 0.0005, -0.0007, 0.0])
    grid = np.zeros((420, 600), np.uint8)
    for y in range(7):
        for x in range(10):
            grid[y*60:(y+1)*60, x*60:(x+1)*60] = 255*((x+y)%2)
    source = np.float32([[0,0],[599,0],[599,419],[0,419]])
    board = np.float32([[0,0,0],[250,0,0],[250,175,0],[0,175,0]])
    rng = np.random.default_rng(10)
    # Map every distorted output pixel to an ideal source pixel.
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    pix = np.stack([xx, yy], -1).astype(np.float32).reshape(-1,1,2)
    xy = cv2.undistortPoints(pix, K, dist, P=K).reshape(h,w,2)
    images = []
    for i in range(14):
        r = rng.uniform(-.32,.32,3)
        t = np.array([-125., -87., 520.]) + rng.uniform([-50,-35,-65],[50,35,90])
        dst, _ = cv2.projectPoints(board, r, t, K, np.zeros(5))
        H = cv2.getPerspectiveTransform(source, dst.reshape(4,2).astype(np.float32))
        ideal = cv2.warpPerspective(grid, H, (w,h), borderValue=170)
        distorted = cv2.remap(ideal, xy[:,:,0], xy[:,:,1], cv2.INTER_LINEAR, borderValue=170)
        images.append(cv2.cvtColor(distorted, cv2.COLOR_GRAY2BGR))
    return images, K, dist


def panel(items, name, cols=3):
    rows = (len(items)+cols-1)//cols
    fig, ax = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), squeeze=False)
    for a, (label, img) in zip(ax.flat, items.items()):
        a.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img, cmap='gray')
        a.set_title(label); a.axis('off')
    for a in list(ax.flat)[len(items):]: a.axis('off')
    fig.tight_layout(); fig.savefig(OUT/name, dpi=120); plt.close(fig)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--image'); p.add_argument('--second'); p.add_argument('--checkerboards'); p.add_argument('--cols',type=int,default=9); p.add_argument('--rows',type=int,default=6); p.add_argument('--square-mm',type=float,default=25); args=p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    img = read_image(args.image) if args.image else synthetic_fruit()
    cv2.imwrite(str(OUT/'input.jpg'),img)
    rng = np.random.default_rng(42); noisy = img.copy()
    salt = rng.random(img.shape[:2]); noisy[salt<.025]=0; noisy[salt>.975]=255
    fs = filters(noisy)
    psnr = {k: float(cv2.PSNR(img,v)) for k,v in {'Noisy':noisy, **fs}.items()}
    panel({'Clean input':img,'Salt and pepper noise':noisy,**fs},'filtering.jpg')
    panel(edges(img),'edges.jpg',2); panel(frequency(img),'frequency.jpg')
    f, counts = features(img); panel(f,'features.jpg')
    second = read_image(args.second) if args.second else cv2.warpAffine(img,cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2),12,0.9),img.shape[1::-1],borderMode=cv2.BORDER_REFLECT)
    matches = {}
    for method in ['ORB','SIFT']:
        vis, matches[method] = match_views(img,second,method); cv2.imwrite(str(OUT/f'{method.lower()}_matches.jpg'),vis)
    if args.checkerboards:
        paths = sorted(p for p in Path(args.checkerboards).iterdir() if p.suffix.lower() in {'.jpg','.png','.jpeg','.bmp'})
        boards = [read_image(p) for p in paths]; trueK=None; trued=None
    else:
        boards,trueK,trued = checkerboards()
    pattern=(args.cols,args.rows) if args.checkerboards else (9,6)
    square=args.square_mm if args.checkerboards else 25
    cal, marked = calibrate(boards,pattern,square)
    b = boards[cal['accepted'][0]]
    panel({'Checkerboard corners':marked,'Distorted view':b,'Undistorted view':undistort(b,cal['K'],cal['distortion'],cal['size'])},'calibration.jpg')
    np.savez(OUT/'calibration.npz',K=cal['K'],dist=cal['distortion'],size=cal['size'])
    fig,ax=plt.subplots(figsize=(6,3)); ax.bar(list(psnr),list(psnr.values())); ax.set_ylabel('PSNR (dB)'); fig.tight_layout(); fig.savefig(OUT/'psnr.png',dpi=120); plt.close(fig)
    result={'image_source':str(args.image) if args.image else 'Synthetic fruit illustration', 'second_source':str(args.second) if args.second else 'Synthetic affine transformation of input', 'calibration_source':'User photographs' if args.checkerboards else 'Synthetic projected checkerboards', 'psnr_db':psnr,'feature_counts':counts,'matches':matches,'calibration':{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in cal.items()},'known_K':None if trueK is None else trueK.tolist(),'known_distortion':None if trued is None else trued.tolist(),'pinhole_example':pinhole([[25,20,500]],cal['K']).tolist()}
    (OUT/'results.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

if __name__=='__main__': main()

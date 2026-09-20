import json, pathlib, subprocess, hashlib
import numpy as np
from PIL import Image, ImageDraw
root=pathlib.Path('evidence/ComfyUI-Sol-H3'); out=pathlib.Path('analysis/results');out.mkdir(parents=True,exist_ok=True)
paths=[root/'MiniMax_H3_00003-audio(2).mp4',root/'MiniMax_H3_00004-audio(5).mp4']
w,h=1216,896
procs=[subprocess.Popen(['ffmpeg','-v','error','-threads','1','-i',str(p),'-f','rawvideo','-pix_fmt','rgb24','-threads','1','-'],stdout=subprocess.PIPE) for p in paths]
xs=[112,192,224,336,384,448,576,704,768,832,960];ys=[128,160,256,320,384,480,512,640]
ratios=[[],[]]; diffmean=np.zeros((h,w)); selected={};n=0
while True:
 raw=[p.stdout.read(w*h*3) for p in procs]
 if not any(raw):break
 assert all(len(r)==w*h*3 for r in raw)
 frames=[np.frombuffer(r,np.uint8).reshape(h,w,3).astype(np.float32)/255 for r in raw]
 for i,a in enumerate(frames):
  gx=np.abs(np.diff(a,axis=1)).mean((0,2));gy=np.abs(np.diff(a,axis=0)).mean((1,2))
  def ratio(g,p):return float(g[p-1]/np.median(np.delete(g[max(0,p-13):p+12],p-1-max(0,p-13))))
  ratios[i].append({'x':[ratio(gx,x) for x in xs],'y':[ratio(gy,y) for y in ys]})
 diffmean+=np.abs(frames[0]-frames[1]).mean(2)
 if n in [0,48,120,174,175,192,240,300]:
  for i,a in enumerate(frames):Image.fromarray(np.uint8(np.clip(a*255,0,255))).save(out/f'v{i+3}_f{n:03}.png')
  selected[n]=frames
 n+=1
for p in procs:assert p.wait()==0
result={'frames':n,'resolution':[w,h],'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'positions':{'x':xs,'y':ys},'formula':'per-frame mean absolute RGB adjacent-line gradient / median of other lines within +/-12 pixels; averaged over frames','groups':{}}
for name,a,b in [('all',0,n),('early_export_frames',0,175),('late_export_frames',175,n)]:
 group={}
 for axis in ['x','y']:
  vals=[np.array([f[axis] for f in r[a:b]]).mean(0) for r in ratios]
  group[axis]={'v3':vals[0].tolist(),'v4':vals[1].tolist(),'delta_v3_minus_v4':(vals[0]-vals[1]).tolist()}
 result['groups'][name]=group
(out/'media_metrics.json').write_text(json.dumps(result,indent=2))
heat=diffmean/n;Image.fromarray(np.uint8(np.clip(heat*255*10,0,255))).save(out/'mean_abs_diff_x10.png')
sheet=Image.new('RGB',(912,3*248));draw=ImageDraw.Draw(sheet)
for row,idx in enumerate([48,174,240]):
 for col,a in enumerate([selected[idx][0],selected[idx][1],np.abs(selected[idx][0]-selected[idx][1])*10]):
  im=Image.fromarray(np.uint8(np.clip(a*255,0,255))).resize((304,224));sheet.paste(im,(col*304,row*248+24));draw.text((col*304+4,row*248+4),f'frame {idx}: '+['v3','v4','abs diff x10'][col],fill='white')
sheet.save(out/'comparison_sheet.jpg')
print(json.dumps(result,indent=2))

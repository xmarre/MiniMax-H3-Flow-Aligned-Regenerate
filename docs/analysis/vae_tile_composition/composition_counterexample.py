"""Execute Core's unchanged composition methods with NumPy tensor primitives.
This probes geometry/arithmetic only; it does not run PyTorch or a learned VAE.
"""
import ast, math, types, json, pathlib
import numpy as np
class Tensor(np.ndarray):
 @property
 def device(self):return 'cpu'
 def clone(self):return self.copy()
 def view(self,*shape):return self.reshape(*shape)
 def copy_(self,other):self[...] = other;return self

def wrap(a):return np.asarray(a).view(Tensor)
torch=types.SimpleNamespace(arange=lambda n,device=None,dtype=None:wrap(np.arange(n,dtype=dtype)),cat=lambda xs,dim:wrap(np.concatenate(xs,axis=dim)),empty=lambda *shape,dtype=None,device=None:wrap(np.empty(shape,dtype=dtype)))
src=pathlib.Path('core/comfy/ldm/minimax/vae.py').read_text();tree=ast.parse(src)
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MiniMaxH3VideoVAE')
methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['split_tiles','blend','tiled_decode']]
mini=ast.Module(body=[ast.ClassDef(name='Core',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])
ns={'torch':torch,'math':math};exec(compile(ast.fix_missing_locations(mini),'<extracted Core methods>','exec'),ns)
Core=ns['Core'];m=Core();m.tile_size=256;m.tile_overlap_min=64;m.vae_ratio=16
# Two rows differing only by a constant, all horizontal tiles identical.
# Exact separable interpolation cannot depend on x in this case.
rows=[[wrap(np.full((1,1,1,256,256),float(i))) for j in range(2)] for i in range(2)]
it=iter(rows);m._decode_tile_row=lambda *args:iter(next(it))
r=m.tiled_decode(wrap(np.zeros((1,24,1,28,28))))
record={'engine':'Core methods extracted via AST; NumPy float64 primitive adapter; no learned decoder', 'source_sha256':__import__('hashlib').sha256(src.encode()).hexdigest(),'two_by_two':{'point_y192_x0':float(r[0,0,0,192,0]),'point_y192_x192':float(r[0,0,0,192,192]),'point_y224_x191':float(r[0,0,0,224,191]),'point_y224_x192':float(r[0,0,0,224,192]),'max_jump_x192':float(np.abs(r[...,192]-r[...,191]).max()),'expected_x_invariance':True}}
assert record['two_by_two']['max_jump_x192']==1
# One-dimensional triple support: decoder constants are 0,1,2.
m.tile_overlap_min=128
starts,lens,ovs=m.split_tiles(1216)
rows=[[wrap(np.full((1,1,1,16,l),float(j))) for j,l in enumerate(lens)]];it=iter(rows);m._decode_tile_row=lambda *args:iter(next(it))
r=m.tiled_decode(wrap(np.zeros((1,24,1,1,76))))
record['triple_overlap']={'starts':starts,'overlaps':ovs,'x223':float(r[0,0,0,0,223]),'x224':float(r[0,0,0,0,224]),'jump':float(r[0,0,0,0,224]-r[0,0,0,0,223])}
# Independent separable overlap-add weight oracle: no learned inference.
def axis_weights(length,ov):
 m.tile_overlap_min=ov;ss,ll,oo=m.split_tiles(length);weights=np.zeros((len(ss),length))
 for i,(s,l) in enumerate(zip(ss,ll)):
  u=np.arange(l,dtype=float);q=np.ones(l)
  if i:q*=np.clip(u/oo[i-1],0,1)
  if i<len(ss)-1:q*=np.clip((l-u)/oo[i],0,1)
  weights[i,s:s+l]=q
 den=weights.sum(0);assert np.all(den>0)
 return weights/den
worst=0.;checks=0
for ov in [64,128,240]:
 for length in range(16,2065,16):
  a=axis_weights(length,ov);worst=max(worst,float(np.abs(a.sum(0)-1).max()));checks+=1
record['weight_oracle']={'cases':checks,'max_partition_error':worst,'positive_denominators':True}
pathlib.Path('analysis/results').mkdir(exist_ok=True)
pathlib.Path('analysis/results/composition_counterexample.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))

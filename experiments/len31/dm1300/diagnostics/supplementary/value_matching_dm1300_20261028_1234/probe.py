
import pathlib,importlib.util,json,math,os,hashlib
import numpy as np
import torch
os.nice(10)
torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(0);torch.set_float32_matmul_precision('highest')
repo=pathlib.Path('/home/qintian/multi-reasoning')
run=repo/'runs/chain_4step_3layer_dm1300_dff2600_len31_6p5m_4x4080_batch4000_compile1_seed2031_eval10000_wd0p15_lr1e-4_fp32_20261027'
spec=importlib.util.spec_from_file_location('model',run/'source_at_launch/model.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
with (run/'latest.pt').open('rb') as f:
 sha=hashlib.sha256(f.read()).hexdigest();f.seek(0);saved=torch.load(f,map_location='cpu',weights_only=False)
model=mod.ReasoningTransformer(**saved['config']['model']).eval();model.load_state_dict(saved['model']);epoch=saved['epoch'];del saved
ex=np.load(repo/'data/chain_4step_6p5m_len31_vocab101_eval10000/eval_x.npy',mmap_mode='r')
ids=np.random.default_rng(20261022).choice(10000,256,replace=False);ids=np.r_[ids,ids+10000];x=np.array(ex[ids],dtype=np.int64)
captured=[]
def hook(module,inputs,output):
 q,k,v=output.view(len(output),31,3,-1).unbind(2);score=q@k.transpose(-2,-1)/math.sqrt(q.size(-1));score.masked_fill_(torch.ones(31,31,dtype=torch.bool).triu(1),-torch.inf);captured.append(score.softmax(-1).numpy())
handles=[b.qkv.register_forward_hook(hook) for b in model.blocks]
arrays=[]
with torch.inference_mode():
 for start in range(0,len(x),16):
  captured.clear();model(torch.from_numpy(x[start:start+16]));arrays.append(np.stack(captured,axis=1))
a=np.concatenate(arrays);assert np.allclose(a.sum(-1),1,atol=2e-6)
arg=a.argmax(-1)
pairs=x[:,:30].reshape(-1,15,2);value=x[:,30].copy();path=[]
for step in range(4):
 hits=pairs[:,:,0]==value[:,None];assert np.all(hits.sum(1)==1);slot=hits.argmax(1);path.append(slot);value=pairs[np.arange(len(x)),slot,1]
on_path=(np.stack(path,axis=1)==0).any(1)


summary={'epoch':epoch,'checkpoint_sha256':sha,'n':len(x),'sampling_seed':20261022,'position_control':'Same query and key positions, same canonical/noncanonical group, unequal value/relationship in other samples; average over observed matched edges.'}
group_ids=np.repeat(np.arange(2),256)
def summarize_edges(edges,mode):
 actual=[];control=[];top=[];control_top=[]
 for b,q,k in edges:
  group=group_ids==group_ids[b]
  if mode=='equal':neg=group&(x[:,q]!=x[:,k])
  elif mode=='predecessor':neg=group&(x[:,q-1]!=x[:,k])
  elif mode=='successor':neg=group&(x[:,q]!=x[:,k-1])
  elif mode=='query_source_at_destination':neg=group&(x[:,q]!=x[:,k-1])
  assert neg.any()
  actual.append(a[b,:,q,k]);control.append(a[neg,:,q,k].mean(0));top.append(arg[b,:,q]==k);control_top.append((arg[neg,:,q]==k).mean(0))
 actual=np.asarray(actual);control=np.asarray(control)
 return {'count':len(edges),'matched_mass':actual.mean(0).tolist(),'position_control_mass':control.mean(0).tolist(),'ratio':(actual.mean(0)/control.mean(0)).tolist(),'top1':np.asarray(top).mean(0).tolist(),'position_control_top1':np.asarray(control_top).mean(0).tolist()}
source_edges=[];dest_edges=[];query_edges=[]
for b in range(len(x)):
 for q in range(1,31):
  for k in np.flatnonzero(x[b,:q]==x[b,q]):
   (query_edges if q==30 else source_edges if q%2==0 else dest_edges).append((b,q,int(k)))
summary['direct_same_value_all_fact_positions']=summarize_edges(source_edges+dest_edges,'equal')
summary['source_to_equal_predecessor_destination']=summarize_edges(source_edges,'equal')
summary['destination_to_equal_successor_source']=summarize_edges(dest_edges,'equal')
summary['final_query_to_same_raw_value']=summarize_edges(query_edges,'equal')
prev_edges=[];next_edges=[];query_value_edges=[]
for b in range(len(x)):
 for j in range(15):
  prev=np.flatnonzero(pairs[b,:,1]==pairs[b,j,0])
  if len(prev) and 2*int(prev[0])+1<2*j+1:prev_edges.append((b,2*j+1,2*int(prev[0])+1))
  follow=np.flatnonzero(pairs[b,:,0]==pairs[b,j,1])
  if len(follow) and 2*int(follow[0])+1<2*j+1:next_edges.append((b,2*j+1,2*int(follow[0])+1))
 first=np.flatnonzero(pairs[b,:,0]==x[b,30]);assert len(first)==1;query_value_edges.append((b,30,2*int(first[0])+1))
summary['destination_to_predecessor_destination']=summarize_edges(prev_edges,'predecessor')
summary['destination_to_successor_destination']=summarize_edges(next_edges,'successor')
summary['final_query_to_first_hop_destination']=summarize_edges(query_value_edges,'query_source_at_destination')
qs=np.arange(2,31)
summary['first_token_attention_excluding_positions1_2']={'mass':a[:,:,qs,0].mean((0,2)).tolist(),'top1':(arg[:,:,qs]==0).mean((0,2)).tolist()}
summary['own_fact_pairing_mass']=a[:,:,np.arange(1,30,2),np.arange(0,30,2)].mean((0,2)).tolist()
for handle in handles:handle.remove()
spec=importlib.util.spec_from_file_location('probe',repo/'diagnostics/live_accuracy_seed2031_dm1300_dff2600/attention_probe.py');probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
with torch.inference_mode():
 t=torch.from_numpy(x[:4]);official=model(t);manual=probe.manual_forward(model,t);torch.testing.assert_close(official,manual,rtol=2e-4,atol=2e-4);summary['reconstruction_max_logit_error']=float((official-manual).abs().max())
output=pathlib.Path(__file__).parent
np.savez_compressed(output/'attention_arrays.npz',attention=a,tokens=x,sample_ids=ids)
(output/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))

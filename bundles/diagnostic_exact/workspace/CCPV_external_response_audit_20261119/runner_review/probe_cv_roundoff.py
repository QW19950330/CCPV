import sys,importlib.util,json,itertools,numpy as np
from pathlib import Path
R=Path(sys.argv[1]);spec=importlib.util.spec_from_file_location('review_runner',R/'run_skillcorner_response.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch,api,ad=m.runtime();P=m.read(R/'skillcorner_response_protocol.json')
rows=[]
for x,y,speed,theta in itertools.product([-52.5,0,52.5],[-34,0,34],[0,.05,1.,5.,10.],[0,np.pi/4,np.pi/2,3*np.pi/4]):
 v=np.array([np.cos(theta),np.sin(theta)],np.float32)*speed
 delta=.1*np.array([-v[1],v[0]])/speed if speed>=.1 else np.array([.1,0],np.float32)
 a=torch.tensor([x,y],dtype=torch.float32);b=a.clone();vv=torch.tensor(v);cmd=vv+torch.tensor(delta,dtype=torch.float32)
 maxerr=0.
 for t in range(1,21):
  a=a+.1*vv;b=b+.1*cmd
  maxerr=max(maxerr,float((b-a-.1*t*(cmd-vv)).abs().max()))
 rows.append({'x':x,'y':y,'speed':speed,'theta':theta,'maxerr':maxerr})
print(json.dumps({'largest':sorted(rows,key=lambda x:-x['maxerr'])[:5],'over1e4':sum(x['maxerr']>1e-4 for x in rows),'total':len(rows)}))

"""Original numerical fragments for the fixed six-fold development replay."""
import numpy as np

NAMES=['mean_abs_x','mean_abs_y','mean_x_squared','mean_y_squared','mean_distance_nearer_goal','min_distance_nearer_goal','n_current_players','n_backward_velocity_players','n_radial_velocity_players','fraction_backward_velocity','fraction_radial_velocity','speed_missing','radial_missing','mean_speed','rms_speed','mean_toward_nearer_goal_velocity']

def objective(w,b,X,y,a):
    z=X@w+b;p=np.exp(-np.logaddexp(0,-z));r=a*(p-y)
    bce=float(a@(np.logaddexp(0,z)-y*z));ridge=float(.005*np.sum(w*w)/2)
    return bce+ridge,X.T@r+.005*w,float(r.sum()),bce,ridge

def normfit(X,a):
    mask=np.isfinite(X);den=(a[:,None]*mask).sum(0);mu=np.divide((a[:,None]*np.where(mask,X,0)).sum(0),den,out=np.zeros(X.shape[1]),where=den>0)
    var=np.divide((a[:,None]*np.where(mask,(X-mu)**2,0)).sum(0),den,out=np.zeros(X.shape[1]),where=den>0)
    sd=np.sqrt(var);sd[sd<1e-12]=1
    return mu,sd

def transform(X,mu,sd):return (np.where(np.isfinite(X),X,mu)-mu)/sd

def fit(X,y,a,initial,bias):
    w=initial.copy();b=float(bias);m=np.zeros(len(w)+1);v=m.copy();history=[]
    for step in range(1,201):
        value,gw,gb,bce,ridge=objective(w,b,X,y,a);g=np.r_[gw,gb];norm=float(np.linalg.norm(g));clip=min(1.,10/max(norm,1e-300));g*=clip
        m=.9*m+.1*g;v=.999*v+.001*g*g;delta=.003*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
        w-=delta[:-1];b-=delta[-1];history.append([step,value,bce,ridge,norm,clip])
    final=objective(w,b,X,y,a);return w,b,m,v,np.array(history),np.array([final[0],np.linalg.norm(np.r_[final[1],final[2]]),final[3],final[4]])

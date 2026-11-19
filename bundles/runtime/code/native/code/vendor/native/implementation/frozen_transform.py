"""Exact old train-frozen scaler application; no fitting or Torch."""
import numpy as np

def transform(X,normalizer):
 X=np.asarray(X,dtype=np.float64);available=np.isfinite(X);z=np.where(available,(X-normalizer['mean'])/normalizer['scale'],0.)
 absent=np.asarray(normalizer['completely_absent_training_feature'],dtype=bool)
 if absent.shape!=(X.shape[1],):raise ValueError('training missing-feature mask shape')
 z[:,absent]=0.
 result=np.concatenate((z,(~available).astype(np.float64)),axis=1)
 if not np.isfinite(result).all():raise ValueError('normalization overflow')
 return result

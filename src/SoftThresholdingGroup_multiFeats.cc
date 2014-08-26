#include "mex.h"
#include <math.h>
#include <string.h>

/*
 * This code is used for group-wise soft thresholding. 
 *  
 * Inputs: 
 *      Vec: the input vector to be shrinkaged (size of N * 1)
 *      Group: the vector denoting the group label of each element in Vec (same size as Vec)
 *      vThresh: the thresholding vectors (size of N_G: number of groups)
 *
 * Output:
 *      Vec_out: the output vector after shrinkage      
 *
 */

// matlab entry point
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{ 
  if (nrhs != 4)
    mexErrMsgTxt("Wrong number of inputs"); 
  if (nlhs != 1)
    mexErrMsgTxt("Wrong number of outputs");

  double* vec, *group, *thresh, *out, *norm, *norm_sk, *mPara ;
  int len_vec, num_groups,k;
  double vnorm, diff, vn;
  const mxArray *mVec = prhs[0];
  const mxArray *mGroup = prhs[1];
  const mxArray *mThresh = prhs[2];
  mPara = mxGetPr(prhs[3]);

  vec = mxGetPr(mVec);
  group = mxGetPr(mGroup);
  thresh = mxGetPr(mThresh);
  
  len_vec = mxGetM(mGroup);  
  num_groups = mPara[0];

  //printf(" %d %d %f", len_vec, num_groups, thresh[0]);

  // output cell
  plhs[0] = mxCreateDoubleMatrix(len_vec, 1, mxREAL);
  out = mxGetPr(plhs[0]);
  
  // do shrinkage
  mxArray* mxNorm = mxCreateDoubleMatrix(num_groups, 1, mxREAL);
  mxArray* mxNorm_sk = mxCreateDoubleMatrix(num_groups, 1, mxREAL);
  norm = mxGetPr(mxNorm);
  norm_sk = mxGetPr(mxNorm_sk);
  
  for(k = 0; k < len_vec; k++)
  {
    int gIdx = group[k];
    if (gIdx > 0)  norm[gIdx - 1] += powf(vec[k], 2); 
  } 

  // shrinkage the norm
  
  for(k = 0; k < num_groups; k++)
  {
     vnorm = sqrt(norm[k]);   
     diff = vnorm - thresh[k];
     norm_sk[k] = (diff > 0) ? diff : 0;
     norm[k] = vnorm;      

    // if (k < 100) printf("(%f %f %d)", norm[k], norm_sk[k], k);
  } 
  printf(" \n");
   
  // compute the shrinked vector
  for(k = 0; k < len_vec; k++)
  {
    int gIdx = group[k];
    if (gIdx > 0)  
    {
        vn = norm[gIdx-1];
        out[k] = (vn > 0) ? vec[k]  * norm_sk[gIdx - 1] / vn : 0;

     //   if (k < 100) printf("(%f %f %f %f %d)", out[k], vec[k], norm_sk[gIdx - 1], vn, k);
    }
    else
    {
        out[k] = vec[k];
    }
  }    
}
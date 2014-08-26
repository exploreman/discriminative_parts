#include "mex.h"
#include <math.h>
#include <string.h>

/*
 * This code is used for computing filter responses.  It computes the
 * response of a set of filters with a feature map.  
 *
 * Basic version, relatively slow but very compatible.
 */

struct thread_data {
  float *A;
  float *B;
  float *M;
  double *C;
  mxArray *mxC;
  const mwSize *A_dims;
  const mwSize *B_dims;
  const mwSize *M_dims;
  int B_scale;
  int num_scale;
  mwSize C_dims[2];
};

// convolve A and B
void process(void *thread_arg) {
  thread_data *args = (thread_data *)thread_arg;
  float *A = args->A;
  float *B = args->B;
  float *M = args->M;
  double *C = args->C;
  const mwSize *A_dims = args->A_dims;
  const mwSize *B_dims = args->B_dims;
  const mwSize *C_dims = args->C_dims; 
  int num_scale = args->num_scale;
  int num_features = args->A_dims[2] / num_scale;
  double *norm = (double *)mxCalloc(C_dims[0]*C_dims[1], sizeof(double));
  int curr_scale = args->B_scale;

  //printf("  %d %d %d\n", num_scale, num_features, curr_scale);

  for (int f = 0; f < num_features; f++) 
  {
    double *dst = C;
    double *dst_norm = norm;
    float *A_src = A + ((curr_scale *  num_features) + f) * A_dims[0]*A_dims[1];      
    float *B_src = B + f;
    float *M_src = M;
   
    for (int x = 0; x < C_dims[1]; x++) 
    {
      for (int y = 0; y < C_dims[0]; y++) 
      {
        double val = 0;
        double val_norm = 0;

        //if (x == 0 & y == 0)
        //{
        //    printf("%f \n", *(M_src + x * C_dims[0] + y) );
       // }

        //if (*(M_src + x * C_dims[0] + y) > 0.5)
        //{
            float *A_off = A_src + x*A_dims[0] + y;
            float *B_off = B_src;
             
             val_norm = pow(*(A_off), 2);
             val = *(A_off++) * *(B_off); 
              
            *(dst++) += val;
            *(dst_norm++) += val_norm;

           // if (x == 0 & y == 0)
           // {
           //     printf("%f %f %f \n", val, val_norm, *(dst-1));
           // }
       //}
       //else
      // {
      //      dst++;
       //     dst_norm++;
       //}
      }
    }
  }
  
 double *dst = C;
 double *dst_norm = norm;
 for (int x = 0; x < C_dims[1]; x++) 
 {
      for (int y = 0; y < C_dims[0]; y++) 
      {
          double val_norm = *(dst_norm++);

          //if ((x == C_dims[1] - 1) && (y == C_dims[0] - 1)) 
          //          printf("debug %f %f \n", val_norm, *(dst));

          if (val_norm != 0)  *(dst) /= sqrt(val_norm);     
          //else   printf("zero-norm \n");

          //if ((x == C_dims[1] - 1) && (y == C_dims[0] - 1)) 
          //          printf("debug %f \n",  *(dst));

          dst++;
      }
 }
 
 mxFree(norm);
  
  //printf("debug: %d %d %d\n", B_dims[0], B_dims[1], num_features);
}

// matlab entry point
// C = fconv(A, cell of B, start, end);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{ 
  if (nrhs != 6)
    mexErrMsgTxt("Wrong number of inputs"); 
  if (nlhs != 1)
    mexErrMsgTxt("Wrong number of outputs");

  // get A
  const mxArray *mxA = prhs[0];
  if (mxGetNumberOfDimensions(mxA) != 3)
    mexErrMsgTxt("Invalid input: A");

  // get B and start/end
  const mxArray *cellB = prhs[1];
  //const mxArray *cellM = prhs[2];
  const mxArray *mScale = prhs[2];
  const mxArray *mNumScale = prhs[5];

  mwSize num_bs = mxGetNumberOfElements(cellB);  
  int start = (int)mxGetScalar(prhs[3]) - 1;
  int end = (int)mxGetScalar(prhs[4]) - 1;
  if (start < 0 || end >= num_bs || start > end)
    mexErrMsgTxt("Invalid input: start/end");
  int len = end-start+1;
    

  // output cell
  plhs[0] = mxCreateCellMatrix(1, len);  // response map
  
  // do convolutions
  thread_data td;
  const mwSize *A_dims = mxGetDimensions(mxA);
  float *A = (float *)mxGetPr(mxA);
  double *Scale = mxGetPr(mScale);
  double* m_num_scale = mxGetPr(mNumScale);
  int num_scale = int(m_num_scale[0]);

  for (int i = 0; i < len; i++) { //len
    const mxArray *mxB = mxGetCell(cellB, i+start);
    //const mxArray *mxM = mxGetCell(cellM, i+start);
    td.A_dims = A_dims;
    td.A = A;
    
    td.B_dims = mxGetDimensions(mxB);
    td.B_scale = int(Scale[i + start]) - 1;
    td.B = (float *)mxGetPr(mxB);
    
    //td.M_dims = mxGetDimensions(mxM);
    //td.M = (float *)mxGetPr(mxM);

    td.num_scale = num_scale;

   
     //if (x == 0 & y == 0)
     {
     //   printf("%f\n", td.B[1]);
     }
 //printf("%d %d %d\n", td.A_dims[2], td.B_dims[1], num_scale);
    if (mxGetNumberOfDimensions(mxB) != 2 ||
        td.A_dims[2] != td.B_dims[1] * num_scale)
    {
        
         mexErrMsgTxt("Invalid input: B");
    }

    // compute size of output
    int height = td.A_dims[0]; // - td.B_dims[0] + 1;
    int width = td.A_dims[1]; // - td.B_dims[1] + 1;
    if (height < 1 || width < 1)
      mexErrMsgTxt("Invalid input: B should be smaller than A");

    td.C_dims[0] = height;
    td.C_dims[1] = width;
    //printf("%d %d %d %d %d %d %d %d %d %d\n", num_scale, len, td.A_dims[0], td.A_dims[1], td.A_dims[2], td.B_dims[1], start, td.B_scale, td.C_dims[0],td.C_dims[2] );


    td.mxC = mxCreateNumericArray(2, td.C_dims, mxDOUBLE_CLASS, mxREAL);
    td.C = (double *)mxGetPr(td.mxC);
    process((void *)&td);
    mxSetCell(plhs[0], i, td.mxC);
  }
}
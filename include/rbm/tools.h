#ifndef __TOOLS_RBM_H__
#define __TOOLS_RBM_H__

#include <math.h>

#include <Eigen/Core>
#include <rbm/types.h>

namespace rbm 
{

  void sampleTwoGaussian(g_float& f1, g_float& f2){
      g_float v1 = (g_float)(rand()+1) / ((g_float)RAND_MAX+1);
      g_float v2 = (g_float)(rand()+1) / ((g_float)RAND_MAX+1);
      g_float len = sqrt(-2.f * log(v1));
      f1 = len * cos(2.f * M_PI * v2);
      f2 = len * sin(2.f * M_PI * v2);
  }


  template <typename T>
    void sampleGaussianMat(T& mat){
        for (int i = 0; i < mat.rows(); ++i){
            int j = 0;
            for ( ; j+1 < mat.cols(); j += 2){
                g_float f1, f2;
                sampleTwoGaussian(f1, f2);
                mat(i,j  ) = f1;
                mat(i,j+1) = f2;
            }
            for (; j < mat.cols(); j ++){
                g_float f1, f2;
                sampleTwoGaussian(f1, f2);
                mat(i, j)  = f1;
            }
        }
    } 

}

#endif

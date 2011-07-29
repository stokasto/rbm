#ifndef __RBM_H__
#define __RBM_H__

#include <math.h>
#include <iostream>
#include <vector>

#include <Eigen/Core>

#include <rbm/types.h>

#define RAND_0_1() ((g_float) rand() / (g_float) RAND_MAX)
#define RAND_MK_K(K) (((RAND_0_1() > 0.5) ? 1. : -1.) * RAND_0_1() * K)

namespace rbm
{
  template<int VIS_DIM, int HID_DIM>
    class RBM
    {
    public:
      /** Type of the Covariance Matrix for a Gaussian of DIM dimensions */
      typedef Eigen::Matrix<g_float, VIS_DIM, HID_DIM> MatrixType;
      /** Type of a Sample Vector for a Gaussian of DIM dimensions */
      typedef Eigen::Matrix<g_float, VIS_DIM, 1> VisVType;
      typedef Eigen::Matrix<g_float, HID_DIM, 1> HidVType;

    private:

      /* for visible layer */
      VisVType visible;
      VisVType visibleact;
      VisVType visiblebias;
      VisVType vbias_deriv;
      /* for hidden layer */
      HidVType hidden;
      HidVType hiddenact;
      HidVType hiddenbias;
      HidVType hbias_deriv;
      HidVType preserved_hidden;
      /* weughts */
      MatrixType weights;
      MatrixType weights_deriv;
      /* general parametrs */
      g_float momentum;
      g_float weights_epsilon;
      g_float cost;

      /* private member functions */
      virtual void
      init_params(g_float eps_w = 0.1f, g_float mom = 0.8f, g_float _cost = 0.0002f)
      {
        weights_epsilon = eps_w;
        momentum = mom;
        cost = _cost;
      }
      virtual void
      activate(LAYER l);

    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      /* constructors */
      RBM() :
        weights_deriv(MatrixType::Zero()), visible(VisVType::Zero()), hidden(HidVType::Zero()),
            visibleact(VisVType::Zero()), hiddenact(HidVType::Zero()),
            vbias_deriv(VisVType::Zero()), hbias_deriv(HidVType::Zero())
      {
        //g_float low   = -4*sqrt(6./(HID_DIM+VIS_DIM));
        g_float high = 4 * sqrt(6. / (HID_DIM + VIS_DIM));

        for (int i = 0; i < weights.rows(); ++i)
          for (int j = 0; j < weights.cols(); ++j)
            {
              weights(i, j) = RAND_MK_K(high);
            }

        // constant bias
        hiddenbias.setConstant(0.5);
        visiblebias.setConstant(0.5);
        /* RANDOM bias
        for (int i = 0; i < HID_DIM; ++i)
          {
            hiddenbias(i) = RAND_MK_K(high);
          }

        for (int i = 0; i < VIS_DIM; ++i)
          {
            visiblebias(i) = RAND_MK_K(high);
          }
        */
        init_params();
      }

      virtual
      ~RBM()
      {
      }

      /* public methods */
      /*virtual bool
       save_to_file(char *fname);
       virtual bool
       load_from_file(char *fname);
       */
      void
      train(int epochs, int num_batches, int cdsteps, bool preserve, std::vector<VisVType> &train_data);
      VisVType
      reconstruct(int epochs);
      VisVType
      reconstruct_from_vector(VisVType &v, int epochs);

      MatrixType &
      getWeights()
      {
        return weights;
      }

      HidVType &
      getHidden()
      {
        return hidden;
      }

      VisVType &
      getVisBias()
      {
        return visiblebias;
      }

      HidVType &
      getHidBias()
      {
        return hiddenbias;
      }

    };
}

#include <rbm/impl/rbm.hpp>

#endif

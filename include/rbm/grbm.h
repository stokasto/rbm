#ifndef __GRBM_H__
#define __GRBM_H__

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
    class GRBM
    {
    public:
      /** Type of the Covariance Matrix for a Gaussian of DIM dimensions */
      typedef Eigen::Matrix<g_float, VIS_DIM, HID_DIM> MatrixType;
      /** Type of a Sample Vector for a Gaussian of DIM dimensions */
      typedef Eigen::Matrix<g_float, VIS_DIM, 1> VisVType;
      typedef Eigen::Matrix<g_float, HID_DIM, 1> HidVType;

    private:

      /* general parameters */
      int numCD, minibatchSize;
      g_float momentum;
      g_float epsStdevs;
      g_float epsW;
      g_float epsVisBias;
      g_float epsHidBias;
      g_float weightCost;
      bool isVisibleBinary;
      bool isVisStdevLearned;
      bool addGaussianNoise;

      /* for visible layer */
      VisVType visibleProb;
      VisVType visiblebias;
      VisVType visiblebiasInc;

      // these are only used if the
      // stdevs are learned from the data
      VisVType visStdevs;
      VisVType visStdevsPos;
      VisVType visStdevsNeg;
      VisVType visStdevsInc;
      // this is used otherwise
      g_float visStdev;

      /* for hidden layer */
      HidVType hiddenProb;
      HidVType hiddenbias;
      HidVType hiddenbiasInc;

      /* weights */
      MatrixType weights;
      MatrixType weightsInc;

      /* cd temporaries */
      int numCases;
      MatrixType cd_weights_tmp;
      VisVType visiblePos;
      VisVType visibleNeg;
      VisVType visStdevsTmp1;
      VisVType visStdevsTmp2;
      HidVType hiddenPos;
      HidVType hiddenNeg;

      /* private member functions */
      virtual void
      init(g_float stdev, g_float eps_weights = 0.00001f, g_float eps_stdev = 0.f,
          g_float mom = 0.9f, g_float weight_cost = 0.f, int num_cd = 1, int batch_size = 100,
          bool vis_binary = false, bool learn_stddev = false, bool add_noise = false)
      {
        // set all learning rates
        epsW = eps_weights;
        epsVisBias = epsW * 10.;
        epsHidBias = epsW * 10.;
        epsStdevs = eps_stdev;

        momentum = mom;
        weightCost = weight_cost;

        numCD = num_cd;
        minibatchSize = batch_size;

        isVisibleBinary = vis_binary;
        isVisStdevLearned = learn_stddev;
        addGaussianNoise = add_noise;

        if (isVisStdevLearned)
          visStdev = 1.;
        else
          visStdev = stdev;

      }

      void
      computeHiddenProb(const VisVType &input);
      void
      computeVisibleProb();
      void
      computeCDPositive(const std::vector<std::pair<VisVType, HidVType> > &data);
      void
      computeCDNegative(const std::vector<std::pair<VisVType, HidVType> > &data);
      void
      updateWeights();

    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      /* constructors */
      GRBM(g_float stdev = 69)
      {
        // init default parameters
        init(stdev);

        // init weights randomly
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

        // constant stdevs in the beginning
        visStdevs.setConstant(stdev);

        // reset all update matrices
        hiddenbiasInc.setZero();
        visiblebiasInc.setZero();
        weightsInc.setZero();
        visStdevsInc.setZero();

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
      }

      virtual
      ~GRBM()
      {
      }

      /* public methods */
      void
      train(const std::vector<VisVType> &train_data, int num_epoch);
      void
      train_epoch(const std::vector<VisVType> &train_data);
      VisVType
      reconstruct(VisVType &v);
    
       
      GRBM &
      setVisibleBinary()
      {
        isVisibleBinary = true;
        return *this;
      }
      
      GRBM &
      setBatchSize(int bsize)
      {
        minibatchSize = bsize;
      }

      GRBM &
      setLearningRate(g_float lrate)
      {  // set all learning rates
        epsW = lrate;
        epsVisBias = epsW * 10.;
        epsHidBias = epsW * 10.;
      }
      
      /*
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
       */
    };
}

#include <rbm/impl/grbm.hpp>

#endif

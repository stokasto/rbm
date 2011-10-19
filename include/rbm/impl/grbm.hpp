#include <rbm/grbm.h>
#include <rbm/tools.h>

#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

using namespace std;

namespace rbm
{

  template<int VIS_DIM, int HID_DIM>
    void
    GRBM<VIS_DIM, HID_DIM>::computeHiddenProb(const VisVType &input)
    {
      // copy input
      visibleProb = input;
      // first check if stddev is also learned
      // if this is the case we need to scale the input data appropriately
      if (isVisStdevLearned)
        {
          visibleProb *= (1. / visStdev);
        }
      // compute activation
      hiddenProb = visibleProb.transpose() * weights;
      if (!isVisibleBinary)
        {
          hiddenProb *= 1. / visStdev;
        }
      hiddenProb += hiddenbias;
      // apply logistic activation
      for (int i = 0; i < hiddenProb.size(); ++i)
        hiddenProb(i) = 1. / (1 + exp(-hiddenProb(i)));
    }

  template<int VIS_DIM, int HID_DIM>
    void
    GRBM<VIS_DIM, HID_DIM>::computeVisibleProb()
    {
      // before we actually compute anything here we need to ensure
      // that the hiddenProbs are binary
      for (int i = 0; i < HID_DIM; ++i)
        {
          hiddenProb(i) = hiddenProb(i) > RAND_0_1() ? 1. : 0.;
        }

      // we have several cases here for which the visible probabilities need to be computed:
      // 1) visible gaussian with learned stdev
      // 2) visible gaussian with added gaussian noise and fixed stdev
      // 3) visible gaussian withoud gaussian noise and fixed stdev
      // 4) visible binary
      if (!isVisibleBinary && isVisStdevLearned)
        {
          visibleProb = weights * hiddenProb;

          if (addGaussianNoise)
            {
              // add some gaussian noise to the visible units here
              g_float gn1, gn2;
              int i = 0;
              for (; i + 1 < visibleProb.size(); i += 2)
                {
                  sampleTwoGaussian(gn1, gn2);
                  visibleProb(i) += gn1;
                  visibleProb(i+1) += gn2;
                }
              for (; i < visibleProb.size(); ++i)
                {
                  sampleTwoGaussian(gn1, gn2);
                  visibleProb(i) += gn1;
                }
            }
          visibleProb.array() = visibleProb.array() * visStdevs.array();
        }
      else
        {
          if (!isVisibleBinary && addGaussianNoise)
            {
              sampleGaussianMat<VisVType>( visibleProb);
              // scale sampled matrix
              //visibleProb *= (1. / visStdev); // << done in next row
              visibleProb.array() += (visibleProb.array() / visStdev)
                  + (weights * hiddenProb).array() / visStdev;
            }
          else
            {
              visibleProb = weights * hiddenProb;
              if (!isVisibleBinary)
                visibleProb *= (1. / visStdev);
            }
        }

      // add bias / prior
      visibleProb += visiblebias;

      // if visible should be binary threshold via logistic activation
      if (isVisibleBinary)
        {
          for (int i = 0; i < visibleProb.size(); ++i)
            visibleProb(i) = 1. / (1 + exp(-visibleProb(i)));
        }
    }

  template<int VIS_DIM, int HID_DIM>
    void
    GRBM<VIS_DIM, HID_DIM>::computeCDPositive(
        const std::vector<std::pair<VisVType, HidVType> > &data)
    {
      assert(data.size() > 0);
      bool computeStdevCD = (isVisStdevLearned && epsStdevs > 0. && !isVisibleBinary);
      numCases = data.size();
      visiblePos.setZero();
      hiddenPos.setZero();
      cd_weights_tmp.setZero();
      if (computeStdevCD)
        { // if we want to learn the stdev we need to clear it first
          visStdevsPos.setZero();
        }

      for (int i = 0; i < data.size(); ++i)
        {
          const std::pair<VisVType, HidVType> &curr = data[i];
          //cd_weights_tmp = visible * hiddenact.transpose();
          cd_weights_tmp += curr.first * curr.second.transpose();
          visiblePos += curr.first;
          hiddenPos += curr.second;

          if (computeStdevCD)
            {
              // TODO: make this efficient and less ugly :/

              // gradient can be computed as:
              // d(-F)/ds_i = ((v_i - b_i)^2 / s_i - v_i * sum_j {h_j * w_ij}) / s_i^2
              visStdevsTmp1 = (weights * curr.second);
              visStdevsTmp1.array() = curr.first.array();
              visStdevsTmp2 = curr.first - visiblebias;
              visStdevsTmp2.array() = (visStdevsTmp2.array() * visStdevsTmp2.array())
                  / visStdevs.array();
              visStdevsTmp2 -= visStdevsTmp1;

              visStdevsPos += visStdevsTmp2;
            }
        }
      weightsInc.array() = weightsInc.array() * momentum + cd_weights_tmp.array() * (epsW
          / numCases);
      //DEBUG:
      //std::cout << "Pos CD weightsInc: " << std::endl << weightsInc << std::endl << std::endl;
      // TODO: Sparsity constraints
    }

  template<int VIS_DIM, int HID_DIM>
    void
    GRBM<VIS_DIM, HID_DIM>::computeCDNegative(
        const std::vector<std::pair<VisVType, HidVType> > &data)
    {
      assert(data.size() > 0);
      bool computeStdevCD = (isVisStdevLearned && epsStdevs > 0. && !isVisibleBinary);
      numCases = data.size();
      visibleNeg.setZero();
      hiddenNeg.setZero();
      cd_weights_tmp.setZero();
      if (computeStdevCD)
        { // if we want to learn the stdev we need to clear it first
          visStdevsNeg.setZero();
        }
      
      //std::cout << "data_size: " << data.size() << std::endl;
      for (int i = 0; i < data.size(); ++i)
        {
          const std::pair<VisVType, HidVType> &curr = data[i];
          //cd_weights_tmp = visible * hiddenact.transpose();
          cd_weights_tmp += curr.first * curr.second.transpose();
          
          //DEBUG:
          /*
          std::cout << "curr.first" << std::endl 
                    << curr.first.transpose() << std::endl;
          std::cout << "cd_weights_tmp" << std::endl
                    << cd_weights_tmp << std::endl << std::endl;
          */
          visibleNeg += curr.first;
          hiddenNeg += curr.second;

          if (computeStdevCD)
            {
              // TODO: make this efficient and less ugly :/

              // gradient can be computed as:
              // d(-F)/ds_i = ((v_i - b_i)^2 / s_i - v_i * sum_j {h_j * w_ij}) / s_i^2
              visStdevsTmp1 = (weights * curr.second);
              visStdevsTmp1.array() = curr.first.array();
              visStdevsTmp2 = curr.first - visiblebias;
              visStdevsTmp2.array() = (visStdevsTmp2.array() * visStdevsTmp2.array())
                  / visStdevs.array();
              visStdevsTmp2 -= visStdevsTmp1;

              visStdevsNeg += visStdevsTmp2;
            }
        }
      weightsInc.array() += cd_weights_tmp.array() * (-epsW / numCases);
      //DEBUG:
      //std::cout << "NEG CD weightsInc: " << std::endl << weightsInc << std::endl << std::endl;
    }

  template<int VIS_DIM, int HID_DIM>
    void
    GRBM<VIS_DIM, HID_DIM>::updateWeights()
    {
      // apply weight costs and momentum
      if (weightCost > 0)
        {
          weightsInc += weights * (-weightCost * epsW / numCases);
        }
      visiblebiasInc = visiblebiasInc.array() * momentum + visiblePos.array() * epsVisBias
          / numCases + visibleNeg.array() * -epsVisBias / numCases;
      hiddenbiasInc = hiddenbiasInc.array() * momentum + hiddenPos.array() * epsHidBias / numCases
          + hiddenNeg.array() * -epsHidBias / numCases;

      // update parameters
      weights += weightsInc;
      visiblebias += visiblebiasInc;
      hiddenbias += hiddenbiasInc;

      if (isVisStdevLearned && epsStdevs > 0.)
        {
          // compute stddev update
          visStdevsPos.array() = visStdevsPos.array() * (epsStdevs / numCases) + visStdevsNeg.array() * (epsStdevs / numCases);
          visStdevsInc = visStdevsInc * momentum + visStdevsPos;

          // and apply
          visStdevs += visStdevsInc;
        }
    }

  template<int VIS_DIM, int HID_DIM>
    void
    GRBM<VIS_DIM, HID_DIM>::train(const std::vector<VisVType> &train_data, int num_epoch)
    {
      int epoch = 1;
      while (epoch <= num_epoch)
        {
          std::cout << "Start training epoch: " << epoch << std::endl;
          train_epoch(train_data);
          std::cout << "Start testing after epoch: " << epoch << std::endl;
          ++epoch;
        }
    }

  template<int VIS_DIM, int HID_DIM>
    void
    GRBM<VIS_DIM, HID_DIM>::train_epoch(const std::vector<VisVType> &train_data)
    {
      int numMinibatches = (train_data.size() + minibatchSize - 1) / minibatchSize;
      double batchError = 0;
      VisVType currentVis = train_data[0];
      std::vector < std::pair<VisVType, HidVType> > cd_data_pos;
      std::vector < std::pair<VisVType, HidVType> > cd_data_neg;
      //cd_data.resize(minibatchSize);
      for (int b = 0; b < numMinibatches; b++)
        {
          int miniStart = b * minibatchSize;
          int miniEnd = std::min(int(train_data.size()), (b + 1) * minibatchSize);
          for (int i = miniStart; i < miniEnd; ++i)
            {
              const VisVType &input = train_data[i];

              // ***** POSITIVE PHASE *****
              computeHiddenProb(input);
              // enqueue to positive examples
              cd_data_pos.push_back(make_pair(input, hiddenProb));

              // ***** NEGATIVE PHASE *****
              // PCD
              if (numCD == 0)
                {
                  computeHiddenProb(currentVis);
                }

              // contrastive divergence
              for (int c = 0; c < numCD; c++)
                {
                  computeVisibleProb();
                  computeHiddenProb(visibleProb);
                  //computeHiddenProb(currentVis);
                }
              // enqueue to negative examples
              cd_data_neg.push_back(make_pair(visibleProb, hiddenProb));
            }

          // compute cd correlations & gradients for positive and negative phase
          computeCDPositive( cd_data_pos);
          computeCDNegative( cd_data_neg);

          // update weights
          updateWeights();

          // for PCD we want to compute the visible activation from the
          // current hidden activation here
          if (numCD == 0)
            computeVisibleProb();

          // and reset cd_data
          cd_data_pos.clear();
          cd_data_neg.clear();
        }
    }

  template<int VIS_DIM, int HID_DIM>
    typename GRBM<VIS_DIM, HID_DIM>::VisVType
    GRBM<VIS_DIM, HID_DIM>::reconstruct(VisVType &v)
    {
      visibleProb = v;
      computeHiddenProb(visibleProb);

      for (int n = 0; n < std::max(1, numCD); n++)
        {
          computeVisibleProb();
          if (n < numCD - 1)
            computeHiddenProb(visibleProb);
        }
      return visibleProb;
    }

}


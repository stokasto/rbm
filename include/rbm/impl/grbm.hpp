#include <rbm/grbm.h>

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
      hiddenProb.array() = 1 / (1 + exp(-hiddenProb.array()));
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
              // TODO add some gaussian noise to the visible units here
            }
          visibleProb.array() = visibleProb.array() * visStdevs.array();
        }
      else
        {
          if (!isVisbleBinary && addGaussianNoise)
            {
              sampleGaussian( visibleProb);
              // scale sampled matrix
              //visibleProb *= (1. / visStdev);
              visbleProb.array() += (visibleProb.array() / visStdev)
                  + (weights * hiddenProb).array() / visStdev;
            }
          else
            {
              visibleProb = weights * hiddenProb;
              if (!isVisbleBinary)
                visibleProb *= (1. / visStdev);
            }
        }

      // add bias / prior
      visibleProb += visiblebias;

      // if visible should be binary threshold via logistic activation
      if (isVisbleBinary)
        {
          visibleProb.array() = 1 / (1 + exp(-visibleProb.array()));
        }
    }

  template<int VIS_DIM, int HID_DIM>
    void
    GRBM<VIS_DIM, HID_DIM>::computeCDPositive(
        const std::vector<std::pair<VisVType, HidVType> > &data)
    {
      bool computeStdevCD = (isVisStdevLearned && epsStdevs > 0. && !isVisibleBinary);
      numCases = data.size();
      visiblePos.setZero();
      hiddenPos.setZero();
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
              visStdevsTmp1 = (weights.transpose() * curr.second);
              visStdevsTmp1.array() = curr.first.array();
              visStdevsTmp2 = curr.first - visiblebias;
              visStdevsTmp2.array() = (visStdevsTmp2.array() * visStdevsTmp2.array())
                  / visStdevs.array();
              visStdevsTmp2 -= visStdevsTmp1;

              visStdevsPos += visStdevsTmp2;
            }
        }
      weightsInc.array() = weightsInc.array() * momentum + cd_weights_pos.array() * (epsW
          / numCases);
      // TODO: Sparsity constraints
    }

  template<int VIS_DIM, int HID_DIM>
    void
    GRBM<VIS_DIM, HID_DIM>::computeCDNegative(
        const std::vector<std::pair<VisVType, HidVType> > &data)
    {
      bool computeStdevCD = (isVisStdevLearned && epsStdevs > 0. && !isVisibleBinary);
      numCases = data.size();
      visibleNeg.setZero();
      hiddenNeg.setZero();
      if (computeStdevCD)
        { // if we want to learn the stdev we need to clear it first
          visStdevsNeg.setZero();
        }

      for (int i = 0; i < data.size(); ++i)
        {
          const std::pair<VisVType, HidVType> &curr = data[i];
          //cd_weights_tmp = visible * hiddenact.transpose();
          cd_weights_tmp += curr.first * curr.second.transpose();
          visibleNeg += curr.first;
          hiddenNeg += curr.second;

          if (computeStdevCD)
            {
              // TODO: make this efficient and less ugly :/

              // gradient can be computed as:
              // d(-F)/ds_i = ((v_i - b_i)^2 / s_i - v_i * sum_j {h_j * w_ij}) / s_i^2
              visStdevsTmp1 = (weights.transpose() * curr.second);
              visStdevsTmp1.array() = curr.first.array();
              visStdevsTmp2 = curr.first - visiblebias;
              visStdevsTmp2.array() = (visStdevsTmp2.array() * visStdevsTmp2.array())
                  / visStdevs.array();
              visStdevsTmp2 -= visStdevsTmp1;

              visStdevsNeg += visStdevsTmp2;
            }
        }
      weightsInc.array() += cd_weights_pos.array() * (-epsW / numCases);
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
          visStdevsPos.array() = visStdevsPos.array() * (epsStdevs / numCases) + visStdevsNeg
              * (epsStdevs / numCases);
          visStdevsInc = visStdevsInc * momentum + visStdevsPos;

          // and apply
          visStdevs += visStdevInc;
        }
    }

  template<int VIS_DIM, int HID_DIM>
    void
    GRBM<VIS_DIM, HID_DIM>::train_epoch(std::vector<VisVType> &train_data, int num_epoch)
    {
      int epoch = 1;
      while (epoch <= num_epoch)
        {
          std::cout << "Start training epoch: " << epoch << std::endl;
          train(train_data);
          std::cout << "Start testing after epoch: " << epoch << std::endl;
          ++epoch;
        }
    }

  template<int VIS_DIM, int HID_DIM>
    void
    GRBM<VIS_DIM, HID_DIM>::train_epoch(std::vector<VisVType> &train_data)
    {
      int numMinibatches = (data.size() + minibatchSize - 1) / minibatchSize;
      double batchError = 0;
      VisVType currentVis = train_data[0];
      std::vector < std::pair<VisVType, HidVType> > cd_data_pos;
      std::vector < std::pair<VisVType, HidVType> > cd_data_neg;
      //cd_data.resize(minibatchSize);
      for (int b = 0; b < numMinibatches; b++)
        {
          int miniStart = m * minibatchSize;
          int miniEnd = std::min(data.size(), (b + 1) * minibatchSize);
          for (int i = miniStart; i < miniEnd; ++i)
            {
              const VisVType &input = train_data[i];

              // ***** POSITIVE PHASE *****
              computeHiddenProb(input);
              // enqueue to positive examples
              cd_data_pos.push_back(make_pair(input, hiddenProbs));

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
                  computeHiddenProb(currentVis);
                }
              // enqueue to negative examples
              cd_data_pos.push_back(make_pair(currentVis, hiddenProbs));
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
    typename RBM<VIS_DIM, HID_DIM>::VisVType
    GRBM<VIS_DIM, HID_DIM>::reconstruct(VisVType &v)
    {
      visibleProbs = v;
      computeHiddenProb( visibleProbs);

      for (int n = 0; n < std::max(1, numCD); n++)
        {
          computeVisibleProb();
          if (n < numCD - 1)
            computeHiddenProb(visibleProbs);
        }
      return visibleProb;
    }

}


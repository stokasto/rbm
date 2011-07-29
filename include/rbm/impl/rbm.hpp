#include <rbm/rbm.h>

#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

using namespace std;

namespace rbm
{

  template<int VIS_DIM, int HID_DIM>
    void
    RBM<VIS_DIM, HID_DIM>::activate(LAYER l)
    {
      switch (l)
        {
      case HIDDEN_LAYER:
        hiddenact = visible.transpose() * weights;
        hiddenact += hiddenbias;
        for (int i = 0; i < HID_DIM; i++)
          {
            hiddenact(i) = act_fun(hiddenact(i), 1.);
            if (hiddenact(i) >= random_uniform_0_1())
              hidden(i) = 1.;
            else
              hidden(i) = 0.;
          }
        break;
      case VISIBLE_LAYER:
        visibleact = weights * hidden;
        visibleact += visiblebias;
        for (int i = 0; i < VIS_DIM; i++)
          {
            visibleact(i) = act_fun(visibleact(i), 1.);
            /*
	    if (visibleact(i) >= random_uniform_0_1())
              visible(i) = 1.;
            else
              visible(i) = 0.;
            */
            visible(i) = visibleact(i);
          }
        break;
      default:
        break;
        }
    }

  template<int VIS_DIM, int HID_DIM>
    typename RBM<VIS_DIM, HID_DIM>::VisVType
    RBM<VIS_DIM, HID_DIM>::reconstruct(int epochs)
    {
      // to compute a reconstruction we initialize with random data
      // and let the RBM run for n epochs
      for (int i = 0; i < VIS_DIM; ++i)
        {
          visible(i) = (random_uniform_0_1() > 0.5) ? 1. : 0.;
        }
      for (int n = 0; n < epochs; n++)
        {
          activate(HIDDEN_LAYER);
          activate(VISIBLE_LAYER);
        }
      return visible;
    }

  template<int VIS_DIM, int HID_DIM>
    typename RBM<VIS_DIM, HID_DIM>::VisVType
    RBM<VIS_DIM, HID_DIM>::reconstruct_from_vector(VisVType &v, int epochs)
    {
      visible = v;
      for (int n = 0; n < epochs; n++)
        {
          activate(HIDDEN_LAYER);
          activate(VISIBLE_LAYER);
        }
      return visible;
    }

  template<int VIS_DIM, int HID_DIM>
    void
    RBM<VIS_DIM, HID_DIM>::train(int epochs, int batch_size, int cdsteps, bool preserve, std::vector<VisVType> &train_data)
    {
      int num_train_data = train_data.size();
      //int update_epoch = (num_train_data - 1) / num_batches;
      float mom;
      // generate matrices that store the cd gradients
      MatrixType cd_weights_pos(MatrixType::Zero());
      MatrixType cd_weights_neg(MatrixType::Zero());
      VisVType visibleact_pos(VisVType::Zero());
      VisVType visibleact_neg(VisVType::Zero());
      HidVType hiddenact_pos(HidVType::Zero());
      HidVType hiddenact_neg(HidVType::Zero());
      for (int e = 0; e < epochs; e++)
        {
          // adaptive cd stepping
          //cdsteps = e % 500 + 1;
          double err_epoch = 0.f;
          if (e < 10)
            { // for the first steps we apply a smaller momentum
              mom = 0.5;
            }
          else
            {
              mom = momentum;
            }
          err_epoch = 0.;
          for (int i = 0; i < num_train_data; i++)
            { // iterate over all training patterns
              VisVType &curr = train_data[i];
              // set visible activation to be the training pattern
              visible = curr;
              /*+++++++++++++++++++++++++++*/
              // now we do what is commonly called the positive phase
              // this means we use the given input data and compute
              // -> hidden activation
              activate(HIDDEN_LAYER);
              // accumulate positive cd by calculating visible * hidden^T
              // updates will later be performed per epoch
              cd_weights_pos += visible * hiddenact.transpose();
              visibleact_pos.array() += visible.array() * visible.array();
              hiddenact_pos.array()  += hiddenact.array() * hiddenact.array();
              if (preserve && (e > 0 || i > 0))
	        { // if we want to use PCD instead of cd grab the preserved activation
		  hidden = preserved_hidden;
		}
              /*+++++++++++++++++++++++++++*/
              // the next step is the so called negative phase
              // hence we use the fantasy output that will be generated
              // by our hidden neurons and use it as the visible input
              // for another computation of the hidden neurons
              for (int n = 0; n < cdsteps; n++)
                { // do n steps of gibbs sampling
		  // --> first compute fantasy
                  activate(VISIBLE_LAYER);
                  // --> next compute hidden activation for fantasy
                  activate(HIDDEN_LAYER);
		  // --> set hidden values to activation rather than samples
		  //     when generating from fantasy --> this will reduce noise in learning
		  //hidden = hiddenact;
                }
	      preserved_hidden = hidden;
              // accumulate negative cd by calculating visible * hidden^T
              // for the given fantasy
              cd_weights_neg += visible * hiddenact.transpose();
              visibleact_neg.array() += visible.array() * visible.array();
              hiddenact_neg.array()  += hiddenact.array() * hiddenact.array();
              // this is just for statistics
              err_epoch += (curr - visible).norm();

              if (i != 0 && i % batch_size == 0)
                {
                  // end of batch --> update parameters
                  // update weights
                  double dW = 0.f;
                  for (int j = 0; j < weights.rows(); ++j)
                    {
                      for (int k = 0; k < weights.cols(); ++k)
                        {
                          dW = weights_deriv(j, k) * mom + weights_epsilon * ((cd_weights_pos(j, k)
                              - cd_weights_neg(j, k)) / batch_size - cost * weights(j, k));
                          //dW = weights_epsilon * ((cd_weights_pos(i, j)
                          //                              - cd_weights_neg(i, j)) / num_train_data - cost * weights(i, j));
                          weights(j, k) += dW; // update weights
                          weights_deriv(j, k) = dW; // save derivative
                        }
                    }
                  // update visible bias
                  for (int j = 0; j < VIS_DIM; ++j)
                    {
                      dW = vbias_deriv(j) * mom + weights_epsilon * ((visibleact_pos(j)
                          - visibleact_neg(j)) / batch_size - cost * visiblebias(j));
                      visiblebias(j) += dW; // update weights
                      vbias_deriv(j) = dW; // save derivative
                    }

                  // update hidden bias
                  for (int j = 0; j < HID_DIM; ++j)
                    {
                      dW = hbias_deriv(j) * mom + weights_epsilon * ((hiddenact_pos(j)
                          - hiddenact_neg(j)) / batch_size - cost * hiddenbias(j));
                      hiddenbias(j) += dW; // update weights
                      hbias_deriv(j) = dW; // save derivative
                    }

                  // clear contrastive divergences
                  cd_weights_pos.setZero();
                  cd_weights_neg.setZero();
                  // clear temporary activations
                  visibleact_pos.setZero();
                  visibleact_neg.setZero();
                  hiddenact_pos.setZero();
                  hiddenact_neg.setZero();
                }
            }
          //std:: cout << "weights after epoch: " << e << " " << weights << std::endl;
          std::cout << "mean error after epoch " << e << " : " << err_epoch / num_train_data
              << std::endl;
        }
    }

}


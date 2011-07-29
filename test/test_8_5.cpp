#include <rbm/rbm.h>

#include <fstream>
#include <vector>
#include <Eigen/Core>

using std::ifstream;
using std::ofstream;
using namespace rbm;

int
main(void)
{
  srand(time(0));
#if 1
  RBM<8,16> r;
  int size = 8;
  ifstream data_in;
  
  std::vector< RBM<8,6>::VisVType > traindata;
  for ( int i = 0; i < size; i++ ) 
    {
    RBM<8,6>::VisVType tmp;
      int pos = i; // / 100;
      for ( int j = 0; j < 8; j++ ) 
        {
          if ( j == pos ) 
            tmp(j) = 1.0f;
          else
            tmp(j) = 0.f;
        }
      std::cout << tmp.transpose() << std::endl;
      traindata.push_back(tmp);
    }

  std::cout << "STARTING TRAIN TEST" << std::endl;
  r.train(1000, 1, 1, false, traindata);
  std::cout << "STARTING REPRODUCTION" << std::endl;
  for ( int i = 0; i < 8; i++ )
    {
      RBM<8,6>::VisVType in(RBM<8,6>::VisVType::Zero());
      in(i) = 1.;
      std::cout << "reconstructing from: " << std::endl << in.transpose() << std::endl;
      RBM<8,6>::VisVType result = r.reconstruct_from_vector(in, 20);
       
      for (int j = 0; j < 8; ++j)
        {
          printf("%f ", result(j));
        }
      printf("\n");
      
      //std::cout << result.transpose() << std::endl;
    }
  std::cout << "weights: " << std::endl << r.getWeights() << std::endl;
  std::cout << "visible bias: " << std::endl << r.getVisBias() << std::endl;
  std::cout << "hidden bias: " << std::endl << r.getHidBias() << std::endl;
#endif
  return 0;
}

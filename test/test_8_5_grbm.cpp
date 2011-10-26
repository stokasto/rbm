#include <rbm/grbm.h>

#include <fstream>
#include <vector>
#include <Eigen/Core>
#include <Eigen/StdVector>

using std::ifstream;
using std::ofstream;
using namespace rbm;

int
main(void)
{
  srand(time(0));
#if 1
  GRBM<8,6> r(4);
  r.setVisibleBinary();
  //r.setStdev(2);
  r.setBatchSize(10);
  r.setLearningRate(0.1);
  int size = 8;
  ifstream data_in;
  
  std::vector< GRBM<8,6>::VisVType, Eigen::aligned_allocator<GRBM<8,6>::VisVType> > traindata;
  for ( int i = 0; i < size; i++ ) 
    {
    GRBM<8,6>::VisVType tmp;
      int pos = i; // / 100;
      for ( int j = 0; j < 8; j++ ) 
        {
          if ( j == pos ) 
            tmp(j) = 1.;//0.5f;
          else
            tmp(j) = 0.;//-0.5f;
        }
      std::cout << tmp.transpose() << std::endl;
      traindata.push_back(tmp);
    }

  std::cout << "STARTING TRAIN TEST" << std::endl;
  r.train(traindata, 500);
  std::cout << "STARTING REPRODUCTION" << std::endl;
  for ( int i = 0; i < 8; i++ )
    {
      GRBM<8,6>::VisVType in(GRBM<8,6>::VisVType::Zero());
      //in.setConstant(-0.5);
      in(i) = 1.;//0.5;
      std::cout << "reconstructing from: " << std::endl << in.transpose() << std::endl;
      GRBM<8,6>::VisVType result = r.reconstruct(in);
       
      for (int j = 0; j < 8; ++j)
        {
          printf("%f ", result(j));
        }
      printf("\n");
      
      //std::cout << result.transpose() << std::endl;
    }
  /*
  std::cout << "weights: " << std::endl << r.getWeights() << std::endl;
  std::cout << "visible bias: " << std::endl << r.getVisBias() << std::endl;
  std::cout << "hidden bias: " << std::endl << r.getHidBias() << std::endl;
  */
#endif
  return 0;
}

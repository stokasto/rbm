/*
 * types.h
 *
 *  Created on: 01.07.2011
 *      Author: springj
 */

#ifndef TYPES_H_
#define TYPES_H_

namespace rbm
{
  enum LAYER
  {
    HIDDEN_LAYER, VISIBLE_LAYER
  };
  typedef float g_float;

  g_float
  random_uniform_0_1()
  {
    return (g_float) rand() / (g_float) RAND_MAX;
  }

  g_float
  act_fun(g_float v, g_float a)
  {
    return 1.f / (1.f + exp(-v * a));
  }
}

#endif /* TYPES_H_ */

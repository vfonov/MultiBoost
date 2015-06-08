#pragma once


#include <vector>
#include <string>
#include <map>
#include <iostream>

namespace MultiBoost
{
  namespace nor_utils{
    class Args;
  }
  
  class GenericStrongLearner;

  /**
  * High level interface to embed MultiBoost in other software
  * 
  * */
  class MultiBoost_HL
  {
  protected:
  public:
      nor_utils::Args *args;
      GenericStrongLearner* pModel;

      MultiBoost_HL();
      ~MultiBoost_HL();
      
      configure
  };

}
#pragma once


#include <vector>
#include <string>
#include <map>
#include <iostream>

/**
 * High level interface to embed MultiBoost in other software
 * 
 * */

namespace MultiBoost
{
  class GenericStrongLearner;

  class MultiBoost_HL
  {
  protected:
  public:
      nor_utils::Args args;
      
      GenericStrongLearner* pModel = NULL;

      MultiBoost_HL();
      ~MultiBoost_HL();
  };

}
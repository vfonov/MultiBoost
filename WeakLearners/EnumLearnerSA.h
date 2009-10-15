/*
* This file is part of MultiBoost, a multi-class 
* AdaBoost learner/classifier
*
* Copyright (C) 2005-2006 Norman Casagrande
* For informations write to nova77@gmail.com
*
* This library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version.
*
* This library is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public
* License along with this library; if not, write to the Free Software
* Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
*
*/

/**
* \file SingleStumpLearner.h A single threshold decision stump learner. 
*/

#ifndef __ENUM_LEARNERSA_H
#define __ENUM_LEARNERSA_H

#include "FeaturewiseLearner.h"
#include "Utils/Args.h"

#include <vector>
#include <fstream>
#include <cassert>

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {

/**
* A \b single threshold decision stump learner. 
* There is ONE and ONE ONLY threshold here.
*/
class EnumLearnerSA : public FeaturewiseLearner
{
public:
   /**
   * Declare weak-learner-specific arguments.
   * adding --baselearnertype
   * \param args The Args class reference which can be used to declare
   * additional arguments.
   * \date 24/04/2007
   */
   virtual void declareArguments(nor_utils::Args& args);

   /**
   * Set the arguments of the algorithm using the standard interface
   * of the arguments. Call this to set the arguments asked by the user.
   * \param args The arguments defined by the user in the command line.
   * \date 24/04/2007
   */
   virtual void initLearningOptions(const nor_utils::Args& args);

   /**
   * The destructor. Must be declared (virtual) for the proper destruction of 
   * the object.
   */
   virtual ~EnumLearnerSA() {}

   /**
   * Returns itself as object.
   * \remark It uses the trick described in http://www.parashift.com/c++-faq-lite/serialization.html#faq-36.8
   * for the auto-registering classes.
   * \date 21/05/2007
   */
   virtual BaseLearner* subCreate() { return new EnumLearnerSA(); }

   /**
   * Run the learner to build the classifier on the given data.
   * \param pData The pointer to the data.
   * \see BaseLearner::run
   * \date 21/05/2007
   */
   virtual float run();
   virtual float run( int colIdx );
   /**
   * Save the current object information needed for classification,
   * that is the _u vector.
   * \param outputStream The stream where the data will be saved
   * \param numTabs The number of tabs before the tag. Useful for indentation
   * \remark To fully save the object it is \b very \b important to call
   * also the super-class method.
   * \date 21/05/2007
   */
   virtual void save(ofstream& outputStream, int numTabs = 0);

   /**
   * Load the xml file that contains the serialized information
   * needed for the classification and that belongs to this class.
   * \param st The stream tokenizer that returns tags and values as tokens
   * \see save()
   * \date 21/05/2007
   */
   virtual void load(nor_utils::StreamTokenizer& st);

   /**
   * Copy all the info we need in classify().
   * pBaseLearner was created by subCreate so it has the correct (sub) type.
   * Usually one must copy the same fields that are loaded and saved. Don't 
   * forget to call the parent's subCopyState().
   * \param pBaseLearner The sub type pointer into which we copy.
   * \see save
   * \see load
   * \see classify
   * \see ProductLearner::run()
   * \date 25/05/2007
   */
   virtual void subCopyState(BaseLearner *pBaseLearner);

   /**
   * The same discriminative function as below, but called with a data point. 
   * \param pData The input data.
   * \param pointIdx The index of the data point.
   * \return \phi[(int)pointIdx]
   * \date 21/05/2007
   */
   virtual float phi(float val, int classIdx) const;

protected:

   vector<float> _u;
   float		 _uOffset;
};

//////////////////////////////////////////////////////////////////////////

} // end of namespace MultiBoost

#endif // __ENUM_LEARNER_H
#pragma warning( disable : 4786 )

#include <vector>
#include <string>
#include <map>
#include <iostream>

#include "Defaults.h"
#include "Utils/Args.h"

#include "StrongLearners/GenericStrongLearner.h"
#include "WeakLearners/BaseLearner.h" // To get the list of the registered weak learners

#include "IO/Serialization.h" // for unserialization

#include "IO/EncodeData.h" // for --encode
#include "IO/InputData.h" // for --encode
#include "WeakLearners/ParasiteLearner.h" // for --encode
#include "StrongLearners/AdaBoostMHLearner.h" // for --encode
#include "StrongLearners/SoftCascadeLearner.h" // for declareBaseArguments
#include "StrongLearners/VJCascadeLearner.h" // for declareBaseArguments
#include "StrongLearners/ArcGVLearner.h" // for declareBaseArguments

#include "IO/OutputInfo.h" // for --encode
#include "Bandits/GenericBanditAlgorithm.h" 

#include "Registrations.h" // for learner and haar features registrations

#include "multiboost_lib.h"

using namespace std;



namespace MultiBoost {

  MultiBoost_HL::MultiBoost_HL()
  {
    pModel = NULL;
    
    // TODO: make this a method
    // initializing the random number generator
    srand ( time(NULL) );
    
    args.setArgumentDiscriminator("--");
        
    args.declareArgument("help");
    args.declareArgument("static");
        
    args.declareArgument("h", "Help", 1, "<optiongroup>");
        
    //////////////////////////////////////////////////////////////////////////
    // Basic Arguments
        
    args.setGroup("Parameters");
    
    args.declareArgument("configfile", "Read some or all the argument from a config file.", 1, "<config file>");
    args.declareArgument("train", "Performs training.", 2, "<dataFile> <nInterations>");
    args.declareArgument("traintest", "Performs training and test at the same time.", 3, "<trainingDataFile> <testDataFile> <nInterations>");
    args.declareArgument("trainvalidtest", "Performs training and test at the same time.", 4, "<trainingDataFile> <validDataFile> <testDataFile> <nInterations>");
    args.declareArgument("test", "Test the model.", 3, "<dataFile> <shypFile> <numIters>");
    args.declareArgument("test", "Test the model and output the results", 4, "<datafile> <shypFile> <numIters> <outFile>");
    args.declareArgument("cmatrix", "Print the confusion matrix for the given model.", 2, "<dataFile> <shypFile>");
    args.declareArgument("cmatrix", "Print the confusion matrix with the class names to a file.", 3, "<dataFile> <shypFile> <outFile>");
    args.declareArgument("posteriors", "Output the posteriors for each class, that is the vector-valued discriminant function for the given dataset and model.", 4, "<dataFile> <shypFile> <outFile> <numIters>");
    args.declareArgument("posteriors", "Output the posteriors for each class, that is the vector-valued discriminant function for the given dataset and model periodically.", 5, "<dataFile> <shypFile> <outFile> <numIters> <period>");    
                
    args.declareArgument("encode", "Save the coefficient vector of boosting individually on each point using ParasiteLearner", 6, "<inputDataFile> <autoassociativeDataFile> <outputDataFile> <nIterations> <poolFile> <nBaseLearners>");   
    args.declareArgument("ssfeatures", "Print matrix data for SingleStump-Based weak learners (if numIters=0 it means all of them).", 4, "<dataFile> <shypFile> <outFile> <numIters>");
        
    args.declareArgument( "fileformat", "Defines the type of intput file. Available types are:\n" 
                          "* simple: each line has attributes separated by whitespace and class at the end (DEFAULT!)\n"
                          "* arff: arff filetype. The header file can be specified using --headerfile option\n"
                          "* arffbzip: bziped arff filetype. The header file can be specified using --headerfile option\n"
                          "* svmlight: \n"
                          "(Example: --fileformat simple)",
                          1, "<fileFormat>" );
        
    args.declareArgument("headerfile", "The header file for arff and SVMLight and arff formats.", 1, "header.txt");
        
    args.declareArgument("constant", "Check constant learner in each iteration.", 0, "");
    args.declareArgument("timelimit", "Time limit in minutes", 1, "<minutes>" );
    args.declareArgument("stronglearner", "Available strong learners:\n"
                         "AdaBoost (default)\n"
                         "FilterBoost\n"
                         "SoftCascade\n"
                         "ArcGV\n"
                         "VJcascade\n", 1, "<stronglearner>" );
        
    args.declareArgument("slowresumeprocess", "Computes every statitstic in each iteration (slow resume)\n"
                         "Computes only the statistics in the last iteration (fast resume, default)\n", 0, "" );
    args.declareArgument("weights", "Outputs the weights of instances at the end of the learning process", 1, "<filename>" );
    args.declareArgument("Cn", "Resampling size for FilterBoost (default=300)", 1, "<value>" );
    args.declareArgument("minmarginthreshold", "Below this margin the coeeficient of weak classifiers are not regularized", 1, "<value>" );
        
    args.declareArgument("onlinetraining", "The weak learner will be trained online\n", 0, "" );
        
    args.declareArgument("earlystopping", "Stop if smoothed test error has not improved for a while.\n"
                         "In traintest mode we may stop before _numIterations iterations or _maxTime time.\n"
                         "In iteration T, if T > <minIterations>, we compute the average test error in the last\n"
                         "<smoothingWindowRate>*T iterations. Let the current minimum of this smoothed error be\n"
                         "taken at Tmin. We stop if T > <maxLookaheadRate>*Tmin.", 
                         3, "<minIterations> <smoothingWindowRate> <maxLookaheadRate>" );

    args.declareArgument("earlystoppingoutputinfo", "Use this outputinfo column instead of the default e01",
      1, "<outputinfoclumn>");

    //// ignored for the moment!
    //args.declareArgument("arffheader", "Specify the arff header.", 1, "<arffHeaderFile>");
        
    // for VJ cascade
    VJCascadeLearner::declareBaseArguments(args);
    
    // for SoftCascade
    SoftCascadeLearner::declareBaseArguments(args);
    //////////////////////////////////////////////////////////////////////////
    // Options
        
    args.setGroup("I/O Options");
        
    /////////////////////////////////////////////
    // these are valid only for .txt input!
    // they might be removed!
    args.declareArgument("d", "The separation characters between the fields (default: whitespaces).\nExample: -d \"\\t,.-\"\nNote: new-line is always included!", 1, "<separators>");
    args.declareArgument("classend", "The class is the last column instead of the first (or second if -examplelabel is active).");
    args.declareArgument("examplename", "The data file has an additional column (the very first) which contains the 'name' of the example.");
    /////////////////////////////////////////////
        
    args.setGroup("Basic Algorithm Options");
    args.declareArgument("weightpolicy", "Specify the type of weight initialization. The user specified weights (if available) are used inside the policy which can be:\n"
                         "* sharepoints Share the weight equally among data points and between positiv and negative labels (DEFAULT)\n"
                         "* sharelabels Share the weight equally among data points\n"
                         "* proportional Share the weights freely", 1, "<weightType>");
        
        
    args.setGroup("General Options");
        
    args.declareArgument("verbose", "Set the verbose level 0, 1 or 2 (0=no messages, 1=default, 2=all messages).", 1, "<val>");
    args.declareArgument("outputinfo", "Output informations on the algorithm performances during training, on file <filename>.", 1, "<filename>");
    args.declareArgument("outputinfo", "Output specific informations on the algorithm performances during training, on file <filename> <outputlist>. <outputlist> must be a concatenated list of three characters abreviation (ex: err for error, fpr for false positive rate)", 2, "<filename> <outputlist>");

    args.declareArgument("seed", "Defines the seed for the random operations.", 1, "<seedval>");
        
    //////////////////////////////////////////////////////////////////////////
    // Shows the list of available learners
    string learnersComment = "Available learners are:";
        
    vector<string> learnersList;
    BaseLearner::RegisteredLearners().getList(learnersList);
    vector<string>::const_iterator it;
    for (it = learnersList.begin(); it != learnersList.end(); ++it)
    {
        learnersComment += "\n ** " + *it;
        // defaultLearner is defined in Defaults.h
        if ( *it == defaultLearner )
            learnersComment += " (DEFAULT)";
    }
        
    args.declareArgument("learnertype", "Change the type of weak learner. " + learnersComment, 1, "<learner>");
        
    //////////////////////////////////////////////////////////////////////////
    //// Declare arguments that belongs to all weak learners
    BaseLearner::declareBaseArguments(args);
        
    ////////////////////////////////////////////////////////////////////////////
    //// Weak learners (and input data) arguments
    for (it = learnersList.begin(); it != learnersList.end(); ++it)
    {
        args.setGroup(*it + " Options");
        // add weaklearner-specific options
        BaseLearner::RegisteredLearners().getLearner(*it)->declareArguments(args);
    }
        
    //////////////////////////////////////////////////////////////////////////
    //// Declare arguments that belongs to all bandit learner
    GenericBanditAlgorithm::declareBaseArguments(args);
  }
  
  

  MultiBoost_HL::~MultiBoost_HL()
  {
    if(pModel)
      delete pModel;
  }

}
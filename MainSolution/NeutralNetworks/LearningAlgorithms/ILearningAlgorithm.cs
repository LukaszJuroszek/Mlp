using NeutralNetworks.Networks;

namespace NeutralNestworks.LearningAlgorithms
{
    interface ILearningAlgorithm
    {
        void Train(INetwork network,double[][] TrainingDataSet,IErrorFunction errorFunction,DataSelection.IDataSelection dataSelection,bool classification,int numEpochs,int batchSize = 0,double learnRate = 0,double momentum = 0,int numRemovedFeatures = 0,double minError = 0,double maxError = 1000);
        double Test(double[][] TrainingDataSet,double[][] TestDataSet);
        bool Cv { get; set; }
        int NumSelectedVectors { get; set; }
    }
}

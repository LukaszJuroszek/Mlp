namespace DL.LearningAlgorithms
{
    interface ILearningAlgorithm
    {
        void Train(Networks.INetwork network,double[][] TrainingDataSet,ErrorFunctions.IErrorFunction errorFunction,DataSelection.IDataSelection dataSelection,bool classification,int numEpochs,int batchSize = 0,double learnRate = 0,double momentum = 0,int numRemovedFeatures = 0,double minError = 0,double maxError = 1000);
        double Test(double[][] TrainingDataSet,double[][] TestDataSet);
        bool cv { get; set; }
        int numSelectedVectors { get; set; }
    }
}

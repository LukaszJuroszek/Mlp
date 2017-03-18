using MLPProgram.Networks;

namespace MLPProgram.LearningAlgorithms
{
    interface ILearningAlgorithm
    {
        void Train(INetwork network,double[][] TrainingDataSet,bool classification,
                                int numEpochs,int batchSize = 30,double learnRate = 0.05,double momentum = 0.5);
        double Test(double[][] TrainingDataSet,double[][] TestDataSet);
    }
}

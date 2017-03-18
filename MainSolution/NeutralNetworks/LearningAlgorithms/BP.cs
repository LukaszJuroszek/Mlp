namespace NeutralNetworks.LearningAlgorithms
{
    class BP : GradientLearning, ILearningAlgorithm
    {
        protected override void UpdateWeights(Networks.MLP network,double learnRate,double momentum,double etaPlus,double etaMinus,double minDelta,double maxDelta,double inputWeightRegularizationCoef = -1)
        {
            for (int L = network.numLayers - 1;L > 0;L--)
                for (int n = 0;n < network.Layer[L];n++)
                    for (int w = 0;w <= network.Layer[L - 1];w++)
                    {
                        network.Weights[L][n][w] += network.weightDiff[L][n][w] + momentum * network.prevWeightDiff[L][n][w];
                        network.prevWeightDiff[L][n][w] = network.weightDiff[L][n][w];
                    }
        }
    }
}






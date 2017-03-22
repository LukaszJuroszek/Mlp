using MLPProgram.Networks;
using Alea;
using Alea.Parallel;
using System;

namespace MLPProgram.LearningAlgorithms
{
    class BP : GradientLearning, ILearningAlgorithm
    {
        [GpuManaged]
        protected override void UpdateWeights(
            MLP network,
            double learnRate,
            double momentum,
            double etaPlus,
            double etaMinus,
            double minDelta,
            double maxDelta,
            double inputWeightRegularizationCoef = -1)
        {
            //var gpu = Gpu.Default;
            for (var l = network._numLayers - 1; l > 0; l--)
            {
                for (var n = 0; n < network._layer[l]; n++)
                {
                    for (var w = 0; w <= network._layer[l - 1]; w++)
                        network._weights[l][n][w] += network._weightDiff[l][n][w] + momentum * network._prevWeightDiff[l][n][w];

                    for (var w = 0; w < network._layer[l - 1]; w++)
                    {
                        network._prevWeightDiff[l][n][w] = network._weightDiff[l][n][w];
                    }
                }
            }
        }
    }
}
}

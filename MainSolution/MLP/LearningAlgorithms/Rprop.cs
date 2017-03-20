using MLPProgram.Networks;
using System;

namespace MLPProgram.LearningAlgorithms
{
    class Rprop : GradientLearning, ILearningAlgorithm
    {
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
            for (var l = network.NumLayers - 1;l > 0;l--)
            {
                for (var n = 0;n < network.Layer[l];n++)
                {
                    for (var w = 0;w <= network.Layer[l - 1];w++)
                    {
                        if (inputWeightRegularizationCoef <= 0 || network.Weights[l][n][w] != 0)
                        {
                            if (network.PrevWeightDiff[l][n][w] * network.WeightDiff[l][n][w] > 0)
                            {
                                network.Delta[l][n][w] *= etaPlus;
                                if (network.Delta[l][n][w] > maxDelta)
                                    network.Delta[l][n][w] = maxDelta;
                            } else if (network.PrevWeightDiff[l][n][w] * network.WeightDiff[l][n][w] < 0)
                            {
                                network.Delta[l][n][w] *= etaMinus;
                                if (network.Delta[l][n][w] < minDelta)
                                    network.Delta[l][n][w] = minDelta;
                            }
                            network.Weights[l][n][w] += Math.Sign(network.WeightDiff[l][n][w]) * network.Delta[l][n][w];
                            network.PrevWeightDiff[l][n][w] = network.WeightDiff[l][n][w];
                        } else
                        {
                            network.PrevWeightDiff[l][n][w] = 0;
                            network.WeightDiff[l][n][w] = 0;
                        }
                    }
                }
            }
        }
    }
}

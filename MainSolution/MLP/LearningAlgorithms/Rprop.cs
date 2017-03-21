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
            for (var l = network._numLayers - 1;l > 0;l--)
            {
                for (var n = 0;n < network._layer[l];n++)
                {
                    for (var w = 0;w <= network._layer[l - 1];w++)
                    {
                        if (inputWeightRegularizationCoef <= 0 || network._weights[l][n][w] != 0)
                        {
                            if (network._prevWeightDiff[l][n][w] * network._weightDiff[l][n][w] > 0)
                            {
                                network._delta[l][n][w] *= etaPlus;
                                if (network._delta[l][n][w] > maxDelta)
                                    network._delta[l][n][w] = maxDelta;
                            } else if (network._prevWeightDiff[l][n][w] * network._weightDiff[l][n][w] < 0)
                            {
                                network._delta[l][n][w] *= etaMinus;
                                if (network._delta[l][n][w] < minDelta)
                                    network._delta[l][n][w] = minDelta;
                            }
                            network._weights[l][n][w] += Math.Sign(network._weightDiff[l][n][w]) * network._delta[l][n][w];
                            network._prevWeightDiff[l][n][w] = network._weightDiff[l][n][w];
                        } else
                        {
                            network._prevWeightDiff[l][n][w] = 0;
                            network._weightDiff[l][n][w] = 0;
                        }
                    }
                }
            }
        }
    }
}

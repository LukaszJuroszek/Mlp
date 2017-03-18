using System;

namespace DL.LearningAlgorithms
{
    class Rprop : GradientLearning, ILearningAlgorithm
    {

        protected override void updateWeights(Networks.MLP network, double learnRate, double momentum, double etaPlus, double etaMinus, double minDelta, double maxDelta, double inputWeightRegularizationCoef = -1)
        {

            for (int L = network.numLayers - 1; L > 0; L--)
                for (int n = 0; n < network.Layer[L]; n++)
                    for (int w = 0; w <= network.Layer[L - 1]; w++)
                    {

                        if (inputWeightRegularizationCoef <= 0 || network.Weights[L][n][w] != 0)
                        {

                            if (network.prevWeightDiff[L][n][w] * network.weightDiff[L][n][w] > 0)
                            {
                                network.Delta[L][n][w] *= etaPlus;
                                if (network.Delta[L][n][w] > maxDelta)
                                    network.Delta[L][n][w] = maxDelta;
                            }
                            else if (network.prevWeightDiff[L][n][w] * network.weightDiff[L][n][w] < 0)
                            {
                                network.Delta[L][n][w] *= etaMinus;
                                if (network.Delta[L][n][w] < minDelta)
                                    network.Delta[L][n][w] = minDelta;
                            }

                            network.Weights[L][n][w] += Math.Sign(network.weightDiff[L][n][w]) * network.Delta[L][n][w];
                            network.prevWeightDiff[L][n][w] = network.weightDiff[L][n][w];
                        }
                        else
                        {
                            network.prevWeightDiff[L][n][w] = 0;
                            network.weightDiff[L][n][w] = 0;
                        }


                    }

        }


    }
}

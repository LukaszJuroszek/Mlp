using MLPProgram.Networks;
using System;

namespace MLPProgram.LearningAlgorithms
{
    class Rprop : GradientLearning, ILearningAlgorithm
    {
        public Rprop(MLP network)
        {
            _network = network;
        }
        protected override void UpdateWeights(
            double learnRate,
            double momentum,
            double etaPlus,
            double etaMinus,
            double minDelta,
            double maxDelta,
            double inputWeightRegularizationCoef = -1)
        {
            for (var l = _network._numLayers - 1;l > 0;l--)
            {
                for (var n = 0;n < _network._layer[l];n++)
                {
                    for (var w = 0;w <= _network._layer[l - 1];w++)
                    {
                        if (inputWeightRegularizationCoef <= 0 || _network._weights[l][n][w] != 0)
                        {
                            if (_network._prevWeightDiff[l][n][w] * _network._weightDiff[l][n][w] > 0)
                            {
                                _network._delta[l][n][w] *= etaPlus;
                                if (_network._delta[l][n][w] > maxDelta)
                                    _network._delta[l][n][w] = maxDelta;
                            } else if (_network._prevWeightDiff[l][n][w] * _network._weightDiff[l][n][w] < 0)
                            {
                                _network._delta[l][n][w] *= etaMinus;
                                if (_network._delta[l][n][w] < minDelta)
                                    _network._delta[l][n][w] = minDelta;
                            }
                            _network._weights[l][n][w] += Math.Sign(_network._weightDiff[l][n][w]) * _network._delta[l][n][w];
                            _network._prevWeightDiff[l][n][w] = _network._weightDiff[l][n][w];
                        } else
                        {
                            _network._prevWeightDiff[l][n][w] = 0;
                            _network._weightDiff[l][n][w] = 0;
                        }
                    }
                }
            }
        }
    }
}

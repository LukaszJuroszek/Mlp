using MLPProgram.Networks;
using Alea;
using Alea.Parallel;
using System;

namespace MLPProgram.LearningAlgorithms
{
    class BP : GradientLearning, ILearningAlgorithm
    {
        public BP(MLP network)
        {
            _network = network;
        }
        //[GpuManaged]
        protected override void UpdateWeights(
            double learnRate,
            double momentum,
            double etaPlus,
            double etaMinus,
            double minDelta,
            double maxDelta,
            double inputWeightRegularizationCoef = -1)
        {
            var gpu = Gpu.Default;
            for (var l = _network._numLayers - 1; l > 0; l--)
            {
                for (var n = 0; n < _network._layer[l]; n++)
                {
                    for (var w = 0; w <= _network._layer[l - 1]; w++)
                        _network._weights[l][n][w] += _network._weightDiff[l][n][w] + momentum * _network._prevWeightDiff[l][n][w];
                    for (var w = 0; w < _network._layer[l - 1]; w++)
                        _network._prevWeightDiff[l][n][w] = _network._weightDiff[l][n][w];
                }
            }
        }
    }
}

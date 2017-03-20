using MLPProgram.Networks;
using Alea;
using Alea.Parallel;

namespace MLPProgram.LearningAlgorithms
{
    class BP : GradientLearning, ILearningAlgorithm
    {
        //[GpuManaged]
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
            //var numLayers = network.NumLayers;
            //int[] layer = network.Layer;
            //double[][][] weights = (double[][][])network.Weights;
            //double[][][] weightDiff = (double[][][])network.WeightDiff;
            //double[][][] prevWeightDiff = (double[][][])network.PrevWeightDiff;
            //var gpu = Gpu.Default;
            for (var l = network.NumLayers - 1; l > 0; l--)
            {
                for (var n = 0; n < network.Layer[l]; n++)
                {
                    for (var w = 0; w <= network.Layer[l - 1]; w++)
                    {
                        network.Weights[l][n][w] += network.WeightDiff[l][n][w] + momentum * network.PrevWeightDiff[l][n][w];
                        network.PrevWeightDiff[l][n][w] = network.WeightDiff[l][n][w];
                    }
                }
            }
        }
    }
}

using MLPProgram.Networks;
using System;

namespace MLPProgram.LearningAlgorithms
{
    abstract class GradientLearning : ILearningAlgorithm
    {
        public double _etaPlus = 1.2, _etaMinus = 0.5, _minDelta = 0.00001, _maxDelta = 10, _errorExponent = 2.0;
        private MLP Network { get; set; }
        public bool cv { get; set; }
        public double Test(double[][] trainingDataSet,double[][] testDataSet)
        {
            return Network.Accuracy(testDataSet, out var errorsRMSE, Network._transferFunction, 0);
        }
        public void Train(INetwork network,double[][] trainingDataSet,
            bool classification,int numEpochs = 30,int batchSize = 30,double learnRate = 0.05,double momentum = 0.5)
        { 
            Network = (MLP)network;
            var numInputs = Network._layer[0];
            var numOutputs = Network._layer[Network._numLayers - 1];
            var numVectors = trainingDataSet.Length;
            if (batchSize > numVectors)
                batchSize = numVectors;
            if (this is Rprop)
                batchSize = numVectors;
            // int maxDegreeOfParallelism = Math.Max(1,(batchSize * network.numWeights) / 250);
            var epoch = 0;
            var derivative = 0.0;
            for (var l = 1;l < Network._numLayers;l++)
                for (var n = 0;n < Network._layer[l];n++)
                    for (var w = 0;w <= Network._layer[l - 1];w++)
                    {
                        Network._weightDiff[l][n][w] = 0;
                        Network._delta[l][n][w] = 0.1;
                    }
            while (epoch < numEpochs) // main training loop
            {
                epoch++;
                MakeGradientZero();
                double sum;
                var v = 0;
                while (v < numVectors)
                {
                    for (var b = 0; b < batchSize; b++)
                    {
                        Network.ForwardPass(trainingDataSet[v], Network._transferFunction);
                        // find SignalErrors for the output layer
                        double sumError = 0;
                        for (var n = 0; n < numOutputs; n++)
                        {
                            var error = trainingDataSet[v][numInputs + n] - Network._output[Network._numLayers - 1][n];
                            error = Math.Sign(error) * Math.Pow(Math.Abs(error), _errorExponent);
                            sumError += Math.Abs(error);
                            if (classification)
                            {
                                derivative = Network._transferFunction.Derivative(Network._output[Network._numLayers - 1][n]);
                            }
                            else
                                derivative = 1.0;
                            Network._signalError[Network._numLayers - 1][n] = error * derivative;
                        }
                        // find SignalErrors for all hidden layers
                        for (var l = Network._numLayers - 2; l > 0; l--)
                            for (var n = 0; n < Network._layer[l]; n++)
                            {
                                sum = 0.0;
                                for (var w = 0; w < Network._layer[l + 1]; w++)
                                    sum += Network._signalError[l + 1][w] * Network._weights[l + 1][w][n];
                                derivative = Network._transferFunction.Derivative(Network._output[l][n]);
                                Network._signalError[l][n] = derivative * sum;
                            }
                        for (var l = Network._numLayers - 1; l > 0; l--)
                            for (var n = 0; n < Network._layer[l]; n++)
                            {
                                //bias
                                Network._weightDiff[l][n][Network._layer[l - 1]] += learnRate * Network._signalError[l][n];
                                for (var w = 0; w < Network._layer[l - 1]; w++)
                                    Network._weightDiff[l][n][w] += learnRate * Network._signalError[l][n] * Network._output[l - 1][w];
                            }
                        v++;
                        if (v == numVectors)
                            break;
                    }
                    UpdateWeights(Network, learnRate, momentum, _etaPlus, _etaMinus, _minDelta, _maxDelta);
                    // zero-out gradients
                    MakeGradientZero();
                }
            }
        }
        private void MakeGradientZero()
        {
            for (var l = 1; l < Network._numLayers; l++)
                for (var n = 0; n < Network._layer[l]; n++)
                    for (var w = 0; w <= Network._layer[l - 1]; w++)
                        Network._weightDiff[l][n][w] = 0;
        }
        abstract protected void UpdateWeights(MLP network,double learnRate,double momentum,double etaPlus,double etaMinus,double minDelta,double maxDelta,double inputWeightRegularizationCoef = -1);
    }
}

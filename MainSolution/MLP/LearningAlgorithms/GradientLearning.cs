using MLPProgram.Networks;
using System;

namespace MLPProgram.LearningAlgorithms
{
    abstract class GradientLearning : ILearningAlgorithm
    {
        public double _etaPlus = 1.2, _etaMinus = 0.5, _minDelta = 0.00001, _maxDelta = 10, _errorExponent = 2.0;
        private MLP Network { get; set; }
        public bool Cv { get; set; }
        public double Test(double[][] trainingDataSet,double[][] testDataSet)
        {
            double errorsRMSE;
            return Network.Accuracy(testDataSet,out errorsRMSE,Network.TransferFunction,0);
        }
        public void Train(INetwork network,double[][] trainingDataSet,
            bool classification,int numEpochs = 30,int batchSize = 30,double learnRate = 0.05,double momentum = 0.5)
        { 
            Network = (MLP)network;
            var numInputs = Network.Layer[0];
            var numOutputs = Network.Layer[Network.NumLayers - 1];
            var numVectors = trainingDataSet.Length;
            if (batchSize > numVectors)
                batchSize = numVectors;
            if (this is Rprop)
                batchSize = numVectors;
            // int maxDegreeOfParallelism = Math.Max(1,(batchSize * network.numWeights) / 250);
            var epoch = 0;
            var derivative = 0.0;
            for (var l = 1;l < Network.NumLayers;l++)
                for (var n = 0;n < Network.Layer[l];n++)
                    for (var w = 0;w <= Network.Layer[l - 1];w++)
                    {
                        Network.WeightDiff[l][n][w] = 0;
                        Network.Delta[l][n][w] = 0.1;
                    }
            while (epoch < numEpochs) // main training loop
            {
                epoch++;
                for (var l = 1;l < Network.NumLayers;l++)
                    for (var n = 0;n < Network.Layer[l];n++)
                        for (var w = 0;w <= Network.Layer[l - 1];w++)
                            Network.WeightDiff[l][n][w] = 0;
                double sum;
                var v = 0;
                while (v < numVectors)
                {
                    for (var b = 0;b < batchSize;b++)
                    {
                        Network.ForwardPass(trainingDataSet[v],Network.TransferFunction);
                        // find SignalErrors for the output layer
                        double sumError = 0;
                        for (var n = 0;n < numOutputs;n++)
                        {
                            var error = trainingDataSet[v][numInputs + n] - Network.Output[Network.NumLayers - 1][n];
                            error = Math.Sign(error) * Math.Pow(Math.Abs(error),_errorExponent);
                            sumError += Math.Abs(error);
                            if (classification)
                            {
                                derivative = Network.TransferFunction.Derivative(Network.Output[Network.NumLayers - 1][n]);
                            } else
                                derivative = 1.0;
                            Network.SignalError[Network.NumLayers - 1][n] = error * derivative;
                        }
                        // find SignalErrors for all hidden layers
                        for (var l = Network.NumLayers - 2;l > 0;l--)
                            for (var n = 0;n < Network.Layer[l];n++)
                            {
                                sum = 0.0;
                                for (var w = 0;w < Network.Layer[l + 1];w++)
                                    sum += Network.SignalError[l + 1][w] * Network.Weights[l + 1][w][n];
                                derivative = Network.TransferFunction.Derivative(Network.Output[l][n]);
                                Network.SignalError[l][n] = derivative * sum;
                            }
                        for (var l = Network.NumLayers - 1;l > 0;l--)
                            for (var n = 0;n < Network.Layer[l];n++)
                            {
                                //bias
                                Network.WeightDiff[l][n][Network.Layer[l - 1]] += learnRate * Network.SignalError[l][n];
                                for (var w = 0;w < Network.Layer[l - 1];w++)
                                    Network.WeightDiff[l][n][w] += learnRate * Network.SignalError[l][n] * Network.Output[l - 1][w];
                            }
                        v++;
                        if (v == numVectors)
                            break;
                    }
                    UpdateWeights(Network,learnRate,momentum,_etaPlus,_etaMinus,_minDelta,_maxDelta);
                    // zero-out gradients
                    for (var l = 1;l < Network.NumLayers;l++)
                        for (var n = 0;n < Network.Layer[l];n++)
                            for (var w = 0;w <= Network.Layer[l - 1];w++)
                                Network.WeightDiff[l][n][w] = 0;
                }
            }
        }
        abstract protected void UpdateWeights(MLP network,double learnRate,double momentum,double etaPlus,double etaMinus,double minDelta,double maxDelta,double inputWeightRegularizationCoef = -1);
    }
}

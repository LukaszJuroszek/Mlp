using Alea;
using MLPProgram.Networks;
using System;

namespace MLPProgram.LearningAlgorithms
{
    abstract class GradientLearning : ILearningAlgorithm
    {

        public double _etaPlus = 1.2, _etaMinus = 0.5, _minDelta = 0.00001, _maxDelta = 10, _errorExponent = 2.0;
        protected MLP _network;
        public bool _cv;
        public double Test(double[][] trainingDataSet, double[][] testDataSet)
        {
            return _network.Accuracy(testDataSet, out var errorsRMSE, _network.transferFunction, 0);
        }
        public void Train(double[][] trainingDataSet,
            bool classification, int numEpochs = 30, int batchSize = 30, double learnRate = 0.05, double momentum = 0.5)
        {
            var numInputs = _network.layer[0];
            var numOutputs = _network.layer[_network.numLayers - 1];
            var numVectors = trainingDataSet.Length;
            if (batchSize > numVectors)
                batchSize = numVectors;
            if (this is Rprop)
                batchSize = numVectors;
            // int maxDegreeOfParallelism = Math.Max(1,(batchSize * network.numWeights) / 250);
            var epoch = 0;
            var derivative = 0.0;
            for (var l = 1; l < _network.numLayers; l++)
                for (var n = 0; n < _network.layer[l]; n++)
                    for (var w = 0; w <= _network.layer[l - 1]; w++)
                    {
                        _network.weightDiff[l][n][w] = 0;
                        _network.delta[l][n][w] = 0.1;
                    }
            while (epoch < numEpochs) // main training loop
            {
                epoch++;
                MakeGradientZero();
                var v = 0;
                while (v < numVectors)
                {
                    for (var b = 0; b < batchSize; b++)
                    {
                        _network.ForwardPass(trainingDataSet[v], _network.transferFunction);
                        // find SignalErrors for the output layer
                        double sumError = 0;
                        for (var n = 0; n < numOutputs; n++)
                        {
                            var error = trainingDataSet[v][numInputs + n] - _network.output[_network.numLayers - 1][n];
                            error = Math.Sign(error) * Math.Pow(Math.Abs(error), _errorExponent);
                            sumError += Math.Abs(error);
                            if (classification)
                            {
                                derivative = _network.transferFunction.Derivative(_network.output[_network.numLayers - 1][n]);
                            }
                            else
                                derivative = 1.0;
                            _network.signalError[_network.numLayers - 1][n] = error * derivative;
                        }
                        // find SignalErrors for all hidden layers
                        for (var l = _network.numLayers - 2; l > 0; l--)
                            for (var n = 0; n < _network.layer[l]; n++)
                            {
                               var sum = 0.0;
                                for (var w = 0; w < _network.layer[l + 1]; w++)
                                    sum += _network.signalError[l + 1][w] * _network.weights[l + 1][w][n];
                                derivative = _network.transferFunction.Derivative(_network.output[l][n]);
                                _network.signalError[l][n] = derivative * sum;
                            }
                        for (var l = _network.numLayers - 1; l > 0; l--)
                            for (var n = 0; n < _network.layer[l]; n++)
                            {
                                //bias
                                _network.weightDiff[l][n][_network.layer[l - 1]] += learnRate * _network.signalError[l][n];
                                for (var w = 0; w < _network.layer[l - 1]; w++)
                                    _network.weightDiff[l][n][w] += learnRate * _network.signalError[l][n] * _network.output[l - 1][w];
                            }
                        v++;
                        if (v == numVectors)
                            break;
                    }
                    UpdateWeights(learnRate, momentum, _etaPlus, _etaMinus, _minDelta, _maxDelta);
                    // zero-out gradients
                    MakeGradientZero();
                }
            }
        }
        private void MakeGradientZero()
        {
            for (var l = 1; l < _network.numLayers; l++)
                for (var n = 0; n < _network.layer[l]; n++)
                    for (var w = 0; w <= _network.layer[l - 1]; w++)
                        _network.weightDiff[l][n][w] = 0;
        }
        abstract protected void UpdateWeights( double learnRate, double momentum, double etaPlus, double etaMinus, double minDelta, double maxDelta, double inputWeightRegularizationCoef = -1);
    }
}

using Alea;
using Alea.Parallel;
using MLPProgram.Networks;
using System;

namespace MLPProgram.LearningAlgorithms
{
    public class GradientLearning
    {
        [GpuParam]
        public double _etaPlus = 1.2, _etaMinus = 0.5, _minDelta = 0.00001, _maxDelta = 10, _errorExponent = 2.0;
        [GpuParam]
        public MLP _network;
        private Gpu gpu;
        public GradientLearning(MLP network)
        {
            _network = network;
            gpu = Gpu.Default;
        }
        [GpuManaged]
        public void Train(int numberOFEpochs = 30, int batchSize = 30, double learnRate = 0.05, double momentum = 0.5)
        {
            var numInputs = _network.baseData._numberOfInput;
            var numOutputs = _network.baseData._numberOfOutput;
            var numVectors = _network.baseData._numberOFVectors;
            var derivative = 0.0;
            if (batchSize > numVectors || nameof(UpdateWeightsRprop).Contains("Rprop"))
                batchSize = numVectors;
            CreateWeightZeroAndAsingDeltaValue(0.1);
            for (var epoch = 0; epoch < numberOFEpochs; epoch++)
            {
                Console.WriteLine(epoch);
                MakeGradientZero();
                var v = 0;
                while (v < numVectors)
                {
                    for (var b = 0; b < batchSize; b++)
                    {
                        Program.ForwardPass(_network, _network.baseData._data[v], _network.baseData);
                        // find SignalErrors for the output layer
                        for (var n = 0; n < numOutputs; n++)
                        {
                            var error = _network.baseData._data[v][numInputs + n] - _network.output[_network.numLayers - 1][n];
                            error = Math.Sign(error) * Math.Pow(Math.Abs(error), _errorExponent);
                            derivative = CalculateDerivativeForSignalErrorsInOutputLayer(n);
                            _network.signalError[_network.numLayers - 1][n] = error * derivative;
                        }
                        for (var l = _network.numLayers - 2; l > 0; l--)
                        {
                            for (var n = 0; n < _network.layer[l]; n++)
                                _network.signalError[l][n] = CalculateDerivativeForHiddenLayer(l, n) * SumSignalErrorForHiddenLayer(l, n);
                        }
                        for (var l = _network.numLayers - 1; l > 0; l--)
                        {
                            for (var n = 0; n < _network.layer[l]; n++)
                            {
                                //bias
                                _network.weightDiff[l][n][_network.layer[l - 1]] += learnRate * _network.signalError[l][n];
                                for (var w = 0; w < _network.layer[l - 1]; w++)
                                    _network.weightDiff[l][n][w] += learnRate * _network.signalError[l][n] * _network.output[l - 1][w];
                            }
                        }
                        v++;
                        if (v == numVectors)
                            break;
                    }
                    UpdateWeightsRprop(learnRate, momentum, _etaPlus, _etaMinus, _minDelta, _maxDelta);
                    // zero-out gradients
                    MakeGradientZero();
                }
            }
        }
        private double SumSignalErrorForHiddenLayer(int layer, int hiddenLayerSecondDim)
        {
            var sum = 0.0;
            for (var w = 0; w < _network.layer[layer + 1]; w++)
                sum += _network.signalError[layer + 1][w] * _network.weights[layer + 1][w][hiddenLayerSecondDim];
            return sum;
        }
        private double CalculateDerivativeForHiddenLayer(int layer, int hidenLayeerSecondDim)
        {
            return _network.baseData.DerivativeFunction(_network.output[layer][hidenLayeerSecondDim]);
        }
        private double CalculateDerivativeForSignalErrorsInOutputLayer(int outputSecondDim)
        {
            double derivative;
            if (_network.classification)
                derivative = _network.baseData.DerivativeFunction(_network.output[_network.numLayers - 1][outputSecondDim]);
            else
                derivative = 1.0;
            return derivative;
        }
        [GpuManaged]
        private void CreateWeightZeroAndAsingDeltaValue(double deltaValue)
        {
            var layers = _network.numLayers;
            var lay = _network.layer;
            var weidiff = _network.weightDiff;
            double[][][] delt = _network.delta;
            for (var l = 1; l < _network.numLayers; l++)
                //gpu.For(1, layers, l =>
                // {
                //gpu.For(1, lay[l], n =>
                // {
                for (var n = 0; n < lay[l]; n++)
                    //for (var w = 0; w <= lay[l - 1]; w++)
                    gpu.For(1, lay[l - 1], w =>
                     {
                         //weidiff[l][n][w] = 0;
                         delt[l][n][w] = 5*5;
                     });
                 //});
             //});
        }
        private void MakeGradientZero()
        {
            for (var l = 1; l < _network.numLayers; l++)
                for (var n = 0; n < _network.layer[l]; n++)
                    for (var w = 0; w <= _network.layer[l - 1]; w++)
                        _network.weightDiff[l][n][w] = 0;
        }
        public double Test(double[][] trainingDataSet, double[][] testDataSet)
        {
            return _network.Accuracy(out var errorsRMSE, 0);
        }
        public void UpdateWeightsRprop(
           double learnRate,
           double momentum,
           double etaPlus,
           double etaMinus,
           double minDelta,
           double maxDelta,
           double inputWeightRegularizationCoef = -1)
        {
            for (var l = _network.numLayers - 1; l > 0; l--)
                for (var n = 0; n < _network.layer[l]; n++)
                    for (var w = 0; w <= _network.layer[l - 1]; w++)
                    {
                        if (inputWeightRegularizationCoef <= 0 || _network.weights[l][n][w] != 0)
                        {
                            if (_network.prevWeightDiff[l][n][w] * _network.weightDiff[l][n][w] > 0)
                            {
                                _network.delta[l][n][w] *= etaPlus;
                                if (_network.delta[l][n][w] > maxDelta)
                                    _network.delta[l][n][w] = maxDelta;
                            }
                            else if (_network.prevWeightDiff[l][n][w] * _network.weightDiff[l][n][w] < 0)
                            {
                                _network.delta[l][n][w] *= etaMinus;
                                if (_network.delta[l][n][w] < minDelta)
                                    _network.delta[l][n][w] = minDelta;
                            }
                            _network.weights[l][n][w] += Math.Sign(_network.weightDiff[l][n][w]) * _network.delta[l][n][w];
                            _network.prevWeightDiff[l][n][w] = _network.weightDiff[l][n][w];
                        }
                        else
                        {
                            _network.prevWeightDiff[l][n][w] = 0;
                            _network.weightDiff[l][n][w] = 0;
                        }
                    }
        }
        public void UpdateWeightsBP(
          double learnRate,
          double momentum,
          double etaPlus,
          double etaMinus,
          double minDelta,
          double maxDelta,
          double inputWeightRegularizationCoef = -1)
        {
            for (var l = _network.numLayers - 1; l > 0; l--)
            {
                for (var n = 0; n < _network.layer[l]; n++)
                {
                    for (var w = 0; w <= _network.layer[l - 1]; w++)
                    {
                        _network.weights[l][n][w] += _network.weightDiff[l][n][w] + momentum * _network.prevWeightDiff[l][n][w];
                        _network.prevWeightDiff[l][n][w] = _network.weightDiff[l][n][w];
                    }
                }
            }
        }
    }
}

﻿using Alea;
using Alea.Parallel;
using MLPProgram.Networks;
using System;

namespace MLPProgram.LearningAlgorithms
{
    abstract class GradientLearning : ILearningAlgorithm
    {

        public double _etaPlus = 1.2, _etaMinus = 0.5, _minDelta = 0.00001, _maxDelta = 10, _errorExponent = 2.0;
        protected MLP _network;
        public double Test(double[][] trainingDataSet, double[][] testDataSet)
        {
            return _network.Accuracy(out var errorsRMSE, 0);
        }
        public void Train(int numEpochs = 30, int batchSize = 30, double learnRate = 0.05, double momentum = 0.5)
        {
            var numInputs = _network.dataFileHolder.NumberOfInput;
            var numOutputs = _network.dataFileHolder.NumberOfOutput;
            var numVectors = _network.dataFileHolder.NumberOFVectors;
            var derivative = 0.0;
            if (batchSize > numVectors || this is Rprop)
                batchSize = numVectors;
            CreateWeightZeroAndAsingDeltaValue(0.1);
            for (var epoch = 0; epoch < numEpochs; epoch++)
            {
                MakeGradientZero();
                var v = 0;
                while (v < numVectors)
                {
                    for (var b = 0; b < batchSize; b++)
                    {
                        Program.ForwardPass(_network, _network.dataFileHolder.Data[v], _network.transferFunction);
                        // find SignalErrors for the output layer
                        double sumError = 0;
                        for (var n = 0; n < numOutputs; n++)
                        {
                            var error = _network.dataFileHolder.Data[v][numInputs + n] - _network.output[_network.numLayers - 1][n];
                            error = Math.Sign(error) * Math.Pow(Math.Abs(error), _errorExponent);
                            sumError += Math.Abs(error);
                            derivative = CalculateDerivativeForSignalErrorsInOutputLayer(n);
                            _network.signalError[_network.numLayers - 1][n] = error * derivative;
                        }
                        // find SignalErrors for all hidden layers
                        for (var l = _network.numLayers - 2; l > 0; l--)
                            for (var n = 0; n < _network.layer[l]; n++)
                                _network.signalError[l][n] = CalculateDerivativeForHiddenLayer(l, n) * SumSignalErrorForHiddenLayer(l, n);
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

        private double SumSignalErrorForHiddenLayer(int layer, int hiddenLayerSecondDim)
        {
            var sum = 0.0;
            for (var w = 0; w < _network.layer[layer + 1]; w++)
                sum += _network.signalError[layer + 1][w] * _network.weights[layer + 1][w][hiddenLayerSecondDim];
            return sum;
        }

        private double CalculateDerivativeForHiddenLayer(int layer, int hidenLayeerSecondDim)
        {
            double derivative;
            if (_network.transferFunction.Method.Name.Equals(nameof(SigmoidTransferFunction)))
                derivative = SigmoidDerivative(_network.output[layer][hidenLayeerSecondDim]);
            else
                derivative = HyperbolicDerivative(_network.output[layer][hidenLayeerSecondDim]);
            return derivative;
        }

        private double CalculateDerivativeForSignalErrorsInOutputLayer(int outputSecondDim)
        {
            double derivative;
            if (_network.classification)
            {
                if (_network.transferFunction.Method.Name.Equals(nameof(SigmoidTransferFunction)))
                    derivative = SigmoidDerivative(_network.output[_network.numLayers - 1][outputSecondDim]);
                else
                    derivative = HyperbolicDerivative(_network.output[_network.numLayers - 1][outputSecondDim]);
            }
            else
                derivative = 1.0;
            return derivative;
        }

        private void CreateWeightZeroAndAsingDeltaValue(double      deltaValue)
        {
            for (var l = 1; l < _network.numLayers; l++)
                for (var n = 0; n < _network.layer[l]; n++)
                    for (var w = 0; w <= _network.layer[l - 1]; w++)
                    {
                        _network.weightDiff[l][n][w] = 0;
                        _network.delta[l][n][w] = deltaValue;
                    }
        }

        private void MakeGradientZero()
        {
            for (var l = 1; l < _network.numLayers; l++)
                for (var n = 0; n < _network.layer[l]; n++)
                    for (var w = 0; w <= _network.layer[l - 1]; w++)
                        _network.weightDiff[l][n][w] = 0;
        }
        abstract protected void UpdateWeights(double learnRate, double momentum, double etaPlus, double etaMinus, double minDelta, double maxDelta, double inputWeightRegularizationCoef = -1);
        public static double HyperbolicTransferFunction(double x)
        {
            return Math.Tanh(x);
        }
        public static double HyperbolicDerivative(double x)
        {
            return 1.0 - x * x;
        }
        public static double SigmoidTransferFunction(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }
        public static double SigmoidDerivative(double x)
        {
            return x * (1.0 - x);
        }
    }
}

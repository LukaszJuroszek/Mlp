using Alea;
using MLPProgram.Networks;
using System;

namespace MLPProgram.LearningAlgorithms
{
    public struct GradientLearning
    {
        public double _etaPlus, _etaMinus, _minDelta, _maxDelta, _errorExponent;
        public MLP _network;
        public GradientLearning(MLP network)
        {
            _etaPlus = 1.2;
            _etaMinus = 0.5;
            _minDelta = 0.00001;
            _maxDelta = 10;
            _errorExponent = 2.0;
            _network = network;
        }
        public void Train( int numberOfEpochs = 30, int batchSize = 30, double learnRate = 0.05, double momentum = 0.5)
        {
            //if (batchSize > _network.baseData._numberOFVectors || nameof(UpdateWeightsRprop).Contains("Rprop"))
            batchSize = _network.baseData._numberOFVectors;
            CreateWeightZeroAndAsingDeltaValue(0.1);
            for (int epoch = 0; epoch < numberOfEpochs; epoch++)
            {
                MakeGradientZero();
                for (int v = 0; v < batchSize; v++)
                {
                    Program.ForwardPass(_network, v);
                    for (int l = 0; l < _network.baseData._numberOfOutput; l++)
                        _network.signalError[_network.numbersOfLayers - 1][l] = CalculateSignalErrors(v, l);
                    for (int l = _network.numbersOfLayers - 2; l > 0; l--)
                        for (int n = 0; n < _network.layer[l]; n++)
                            _network.signalError[l][n] = CalculateDerivativeForHiddenLayer(l, n) * SumSignalErrorForHiddenLayer(l, n);
                    for (int l = _network.numbersOfLayers - 1; l > 0; l--)
                        for (int n = 0; n < _network.layer[l]; n++)
                            Bias(learnRate, l, n);
                }
                UpdateWeightsRprop(learnRate, momentum, _etaPlus, _etaMinus, _minDelta, _maxDelta);
                // zero-out gradients
                MakeGradientZero();
            }
        }
        public static int Sign(double number)
        {
            return number > 0 ? 1 : number < 0 ? -1 : 0;
        }
        private double CalculateSignalErrors(int v, int n)
        {
            double error = _network.baseData._trainingDataSet[v][_network.baseData._numberOfInput + n] - _network.output[_network.numbersOfLayers - 1][n];
            error = Math.Sign(error) * Math.Pow(Math.Abs(error), _errorExponent);
            double derivative = CalculateDerivativeForSignalErrorsInOutputLayer(n);
            return error * derivative;
        }

        private void Bias(double learnRate, int l, int n)
        {
            _network.weightDiff[l][n][_network.layer[l - 1]] += learnRate * _network.signalError[l][n];
            for (int w = 0; w < _network.layer[l - 1]; w++)
                _network.weightDiff[l][n][w] += learnRate * _network.signalError[l][n] * _network.output[l - 1][w];
        }
        private double SumSignalErrorForHiddenLayer(int layer, int hiddenLayerSecondDim)
        {
            double sum = 0.0;
            for (int w = 0; w < _network.layer[layer + 1]; w++)
                sum += _network.signalError[layer + 1][w] * _network.weights[layer + 1][w][hiddenLayerSecondDim];
            return sum;
        }
        private double CalculateDerivativeForHiddenLayer(int layer, int hidenLayeerSecondDim)
        {
            return DerivativeFunction(_network.output[layer][hidenLayeerSecondDim]);
        }
        private double CalculateDerivativeForSignalErrorsInOutputLayer(int outputSecondDim)
        {
            double derivative;
            if (_network.classification)
                derivative = DerivativeFunction(_network.output[_network.numbersOfLayers - 1][outputSecondDim]);
            else
                derivative = 1.0;
            return derivative;
        }
        private void CreateWeightZeroAndAsingDeltaValue(double deltaValue)
        {
            for (int l = 1; l < _network.numbersOfLayers; l++)
                for (int n = 0; n < _network.layer[l]; n++)
                {
                    for (int w = 0; w <= _network.layer[l - 1]; w++)
                    {
                        _network.weightDiff[l][n][w] = 0;
                        _network.delta[l][n][w] = deltaValue;
                    }
                }
        }
        private void MakeGradientZero()
        {
            for (int l = 1; l < _network.numbersOfLayers; l++)
                for (int n = 0; n < _network.layer[l]; n++)
                    for (int w = 0; w <= _network.layer[l - 1]; w++)
                        _network.weightDiff[l][n][w] = 0;
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
            for (int l = _network.numbersOfLayers - 1; l > 0; l--)
                for (int n = 0; n < _network.layer[l]; n++)
                    for (int w = 0; w <= _network.layer[l - 1]; w++)
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
            for (int l = _network.numbersOfLayers - 1; l > 0; l--)
            {
                for (int n = 0; n < _network.layer[l]; n++)
                {
                    for (int w = 0; w <= _network.layer[l - 1]; w++)
                    {
                        _network.weights[l][n][w] += _network.weightDiff[l][n][w] + momentum * _network.prevWeightDiff[l][n][w];
                        _network.prevWeightDiff[l][n][w] = _network.weightDiff[l][n][w];
                    }
                }
            }
        }
        public static double TransferFunction(bool isSigmoidFunction, double x)
        {
            double result = 0;
            result = isSigmoidFunction ? SigmoidTransferFunction(x) : HyperbolicTransferFunction(x);
            return result;
        }

        public static double DerivativeFunction(bool isSigmoidFunction, double x)
        {
            double result = 0;
            result = isSigmoidFunction ? SigmoidDerivative(x) : HyperbolicDerivative(x);
            return result;
        }
        public static double TransferFunction(MLPNew _network, double x)
        {
            double result = 0;
            if (_network.baseData._isSigmoidFunction)
                result = SigmoidTransferFunction(x);
            else
                result = HyperbolicTransferFunction(x);
            return result;
        }
        public static double TransferFunction(MLP _network, double x)
        {
            double result = 0;
            if (_network.baseData._isSigmoidFunction)
                result = SigmoidTransferFunction(x);
            else
                result = HyperbolicTransferFunction(x);
            return result;
        }
        public double DerivativeFunction(double x)
        {
            double result = 0;
            if (_network.baseData._isSigmoidFunction)
                result = SigmoidDerivative(x);
            else
                result = HyperbolicDerivative(x);
            return result;
        }
        public static double HyperbolicTransferFunction(double x)
        {
            return DeviceFunction.Tanh(x);
        }
        public static double HyperbolicDerivative(double x)
        {
            return 1.0 - x * x;
        }
        public static double SigmoidTransferFunction(double x)
        {
            return 1.0 / (1.0 + DeviceFunction.Exp(-x));
        }
        public static double SigmoidDerivative(double x)
        {
            return x * (1.0 - x);
        }
        public static bool IsSigmoidTransferFunction(Func<double, double> func)
        {
            return func.Method.Name.Equals("SigmoidTransferFunction") ? true : false;
        }
    }
}

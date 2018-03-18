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
        public void Train(int numberOfEpochs = 30, int batchSize = 30, double learnRate = 0.05, double momentum = 0.5)
        {
            //if (batchSize > _network.baseData._numberOFVectors || nameof(UpdateWeightsRprop).Contains("Rprop"))
            //batchSize = _network.baseData._numberOFVectors;
            CreateWeightZeroAndAsingDeltaValue(_network, 0.1);
            for (int epoch = 0; epoch < numberOfEpochs; epoch++)
            {
                MakeGradientZero(_network);
                CalculateForAllVectors(_network, _errorExponent, learnRate);
                UpdateWeightsRprop(_network, learnRate, momentum, _etaPlus, _etaMinus, _minDelta, _maxDelta);
                // zero-out gradients
                MakeGradientZero(_network);
            }
        }
        public static void CalculateForAllVectors(MLP network, double errorExponent, double learnRate)
        {
            for (int batch = 0; batch < network.baseData._numberOFVectors; batch++)
            {
                Program.ForwardPass(network, batch);
                for (int l = 0; l < network.baseData._numberOfOutput; l++)
                    network.signalError[network.numbersOfLayers - 1][l] = CalculateSignalErrorsForOutputLayer(network, batch, l, errorExponent);
                for (int l = network.numbersOfLayers - 2; l > 0; l--)
                    for (int n = 0; n < network.layer[l]; n++)
                        network.signalError[l][n] = CalculateSignalErrorFroHiddenLayer(network, l, n);
                CalculateBias(network, learnRate);
            }
        }
        public static double CalculateSignalErrorFroHiddenLayer(MLP netwrok, int l, int n)
        {
            return CalculateDerivativeForHiddenLayer(netwrok, l, n) * SumSignalErrorForHiddenLayer(netwrok, l, n);
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
        public static double CalculateSignalErrorsForOutputLayer(MLP network, int v, int n, double errorExponent)
        {
            double error = network.baseData._trainingDataSet[v][network.baseData._numberOfInput + n] - network.output[network.numbersOfLayers - 1][n];
            error = Math.Sign(error) * Math.Pow(Math.Abs(error), errorExponent);
            double derivative = CalculateDerivativeForSignalErrorsInOutputLayer(network, n);
            return error * derivative;
        }
        public static void CalculateBias(MLP network, double learnRate)
        {
            for (int l = network.numbersOfLayers - 1; l > 0; l--)
                for (int n = 0; n < network.layer[l]; n++)
                {
                    network.weightDiff[l][n][network.layer[l - 1]] += learnRate * network.signalError[l][n];
                    for (int w = 0; w < network.layer[l - 1]; w++)
                        network.weightDiff[l][n][w] += learnRate * network.signalError[l][n] * network.output[l - 1][w];
                }
        }
        private double SumSignalErrorForHiddenLayer(int layer, int hiddenLayerSecondDim)
        {
            double sum = 0.0;
            for (int w = 0; w < _network.layer[layer + 1]; w++)
                sum += _network.signalError[layer + 1][w] * _network.weights[layer + 1][w][hiddenLayerSecondDim];
            return sum;
        }
        private static double SumSignalErrorForHiddenLayer(MLP network, int l, int n)
        {
            double sum = 0.0;
            for (int w = 0; w < network.layer[l + 1]; w++)
                sum += network.signalError[l + 1][w] * network.weights[l + 1][w][n];
            return sum;
        }
        private double CalculateDerivativeForHiddenLayer(int l, int n)
        {
            return DerivativeFunction(_network.output[l][n]);
        }
        private static double CalculateDerivativeForHiddenLayer(MLP network, int l, int n)
        {
            return DerivativeFunction(network.classification, network.output[l][n]);
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
        private static double CalculateDerivativeForSignalErrorsInOutputLayer(MLP network, int n)
        {
            double derivative;
            if (network.classification)
                derivative = DerivativeFunction(network.classification, network.output[network.numbersOfLayers - 1][n]);
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
        public static void CreateWeightZeroAndAsingDeltaValue(MLP network, double deltaValue)
        {
            for (int l = 1; l < network.numbersOfLayers; l++)
                for (int n = 0; n < network.layer[l]; n++)
                {
                    for (int w = 0; w <= network.layer[l - 1]; w++)
                    {
                        network.weightDiff[l][n][w] = 0;
                        network.delta[l][n][w] = deltaValue;
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
        public static void MakeGradientZero(MLP network)
        {
            for (int l = 1; l < network.numbersOfLayers; l++)
                for (int n = 0; n < network.layer[l]; n++)
                    for (int w = 0; w <= network.layer[l - 1]; w++)
                        network.weightDiff[l][n][w] = 0;
        }
        public static void UpdateWeightsRprop(
            MLP network,
           double learnRate,
           double momentum,
           double etaPlus,
           double etaMinus,
           double minDelta,
           double maxDelta,
           double inputWeightRegularizationCoef = -1)
        {
            for (int l = network.numbersOfLayers - 1; l > 0; l--)
            {
                for (int n = 0; n < network.layer[l]; n++)
                    for (int w = 0; w <= network.layer[l - 1]; w++)
                    {
                        if (inputWeightRegularizationCoef <= 0 || network.weights[l][n][w] != 0)
                        {
                            if (network.prevWeightDiff[l][n][w] * network.weightDiff[l][n][w] > 0)
                            {
                                network.delta[l][n][w] *= etaPlus;
                                if (network.delta[l][n][w] > maxDelta)
                                    network.delta[l][n][w] = maxDelta;
                            }
                            else if (network.prevWeightDiff[l][n][w] * network.weightDiff[l][n][w] < 0)
                            {
                                network.delta[l][n][w] *= etaMinus;
                                if (network.delta[l][n][w] < minDelta)
                                    network.delta[l][n][w] = minDelta;
                            }
                            network.weights[l][n][w] += Math.Sign(network.weightDiff[l][n][w]) * network.delta[l][n][w];
                            network.prevWeightDiff[l][n][w] = network.weightDiff[l][n][w];
                        }
                        else
                        {
                            network.prevWeightDiff[l][n][w] = 0;
                            network.weightDiff[l][n][w] = 0;
                        }
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
        public static double TransferFunction(byte isSigmoidFunction, double x)
        {
            double result = 0;
            result = isSigmoidFunction == 1 ? SigmoidTransferFunction(x) : HyperbolicTransferFunction(x);
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
            if (_network.baseData._isSigmoidFunction == 1)
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
        public static byte IsSigmoidTransferFunctionByte(Func<double, double> func)
        {
            return func.Method.Name.Equals("SigmoidTransferFunction") ? Convert.ToByte(1) : Convert.ToByte(0);
        }
    }
}

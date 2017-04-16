using Alea;
using MLPProgram.Networks;
using System;

namespace MLPProgram.LearningAlgorithms
{
    public struct GradientLearningNew
    {
        public double _etaPlus, _etaMinus, _minDelta, _maxDelta, _errorExponent;
        public MLPNew _network;
        public GradientLearningNew(MLPNew network)
        {
            _etaPlus = 1.2;
            _etaMinus = 0.5;
            _minDelta = 0.00001;
            _maxDelta = 10;
            _errorExponent = 2.0;
            _network = network;
        }
        public MLPNew Train(int numberOfEpochs = 30, int batchSize = 30, double learnRate = 0.05, double momentum = 0.5)
        {
            double errorExponent = _errorExponent;
            batchSize = _network.baseData._numberOfInputRow;
            CreateWeightZeroAndAsingDeltaValue(_network, 0.1);
            MakeGradientZero(_network);
            for (int epoch = 0; epoch < numberOfEpochs; epoch++)
            {
                for (int batch = 0; batch < batchSize; batch++)
                {
                    MakeGradientZero(_network);
                    Program.ForwardPass(_network, batch);
                    for (int l = 0; l < _network.baseData._numberOfOutput; l++)
                    {
                        double error = _network.baseData._trainingDataSet[batch, _network.baseData._numberOfInput + l] - _network.output[(int)NetworkLayer.Output][l];
                        error = Sign(error) * DeviceFunction.Pow(DeviceFunction.Abs(error), errorExponent);
                        double derivative = _network.classification ? DerivativeFunction(_network.classification, _network.output[(int)NetworkLayer.Output][l]) : 1.0;
                        _network.signalError[(int)NetworkLayer.Output][l] = error * derivative;
                    }
                    for (int l = _network.numbersOfLayers - 2; l > 0; l--)
                        for (int n = 0; n < _network.networkLayers[l]; n++)
                            _network.signalError[l][n] = DerivativeFunction(_network.classification, _network.output[l][n]) * SumSignalErrorForHiddenLayer(_network, l, n);
                    for (int l = _network.numbersOfLayers - 1; l > 0; l--)
                        for (int n = 0; n < _network.networkLayers[l]; n++)
                        {
                            _network.weightDiff[l][n, _network.networkLayers[l - 1]] += learnRate * _network.signalError[l][n];
                            for (int w = 0; w < _network.networkLayers[l - 1]; w++)
                                _network.weightDiff[l][n, w] += learnRate * _network.signalError[l][n] * _network.output[l - 1][w];
                        }
                }
                UpdateWeightsRprop(_network, learnRate, momentum, _etaPlus, _etaMinus, _minDelta, _maxDelta);
                MakeGradientZero(_network);
            }
            return _network;
        }
        public static void AddWeigth(MLPNew first, MLPNew second)
        {
            for (int l = 1; l < first.numbersOfLayers; l++)
            {
                for (int n = 0; n < first.weights[l].GetLength(0); n++)
                {
                    for (int w = 0; w < first.weights[l].GetLength(1); w++)
                    {
                        first.weights[l][n, w] += second.weights[l][n, w];
                    }
                }
            }
        }
        public static void AddOutput(MLPNew first, MLPNew second)
        {
            for (int l = 1; l < first.numbersOfLayers; l++)
            {
                for (int n = 0; n < first.output[l].GetLength(0); n++)
                {
                    first.output[l][n] += second.output[l][n];
                }
            }
        }
        public static int Sign(double number)
        {
            return number > 0 ? 1 : number < 0 ? -1 : 0;
        }
        private static double CalculateSignalErrors(MLPNew network, int v, int n, double errorExponent)
        {
            double error = network.baseData._trainingDataSet[v, network.baseData._numberOfInput + n] - network.output[(int)NetworkLayer.Output][n];
            error = Sign(error) * DeviceFunction.Pow(DeviceFunction.Abs(error), errorExponent);
            double derivative = network.classification ? DerivativeFunction(network.classification, network.output[(int)NetworkLayer.Output][n]) : 1.0;
            return error * derivative;
        }

        private static double SumSignalErrorForHiddenLayer(MLPNew network, int layer, int hiddenLayerSecondDim)
        {
            double sum = 0.0;
            for (int w = 0; w < network.networkLayers[layer + 1]; w++)
                sum += network.signalError[layer + 1][w] * network.weights[layer + 1][w, hiddenLayerSecondDim];
            return sum;
        }
        private static void CreateWeightZeroAndAsingDeltaValue(MLPNew network, double deltaValue)
        {
            for (int l = 1; l < network.numbersOfLayers; l++)
            {
                for (int n = 0; n < network.networkLayers[l]; n++)
                {
                    for (int w = 0; w < network.networkLayers[l - 1]; w++)
                    {
                        network.weightDiff[l][n, w] = 0;
                        network.delta[l][n, w] = deltaValue;
                    }
                }
            }
        }
        private static void MakeGradientZero(MLPNew network)
        {
            for (int l = 1; l < network.numbersOfLayers; l++)
                for (int n = 0; n < network.networkLayers[l]; n++)
                    for (int w = 0; w < network.networkLayers[l - 1]; w++)
                        network.weightDiff[l][n, w] = 0;
        }
        public static void UpdateWeightsRprop(MLPNew network,
           double learnRate,
           double momentum,
           double etaPlus,
           double etaMinus,
           double minDelta,
           double maxDelta,
           double inputWeightRegularizationCoef = -1)
        {
            for (int l = network.numbersOfLayers - 1; l > 0; l--)
                for (int n = 0; n < network.networkLayers[l]; n++)
                    for (int w = 0; w <= network.networkLayers[l - 1]; w++)
                    {
                        if (inputWeightRegularizationCoef <= 0 || network.weights[l][n, w] != 0)
                        {
                            if (network.prevWeightDiff[l][n, w] * network.weightDiff[l][n, w] > 0)
                            {
                                network.delta[l][n, w] *= etaPlus;
                                if (network.delta[l][n, w] > maxDelta)
                                    network.delta[l][n, w] = maxDelta;
                            }
                            else if (network.prevWeightDiff[l][n, w] * network.weightDiff[l][n, w] < 0)
                            {
                                network.delta[l][n, w] *= etaMinus;
                                if (network.delta[l][n, w] < minDelta)
                                    network.delta[l][n, w] = minDelta;
                            }
                            network.weights[l][n, w] += Sign(network.weightDiff[l][n, w]) * network.delta[l][n, w];
                            network.prevWeightDiff[l][n, w] = network.weightDiff[l][n, w];
                        }
                        else
                        {
                            network.prevWeightDiff[l][n, w] = 0;
                            network.weightDiff[l][n, w] = 0;
                        }
                    }
        }
        //public void UpdateWeightsBP(
        //  double learnRate,
        //  double momentum,
        //  double etaPlus,
        //  double etaMinus,
        //  double minDelta,
        //  double maxDelta,
        //  double inputWeightRegularizationCoef = -1)
        //{
        //    for (int l = _network.numbersOfLayers - 1; l > 0; l--)
        //    {
        //        for (int n = 0; n < _network.layer[l]; n++)
        //        {
        //            for (int w = 0; w <= _network.layer[l - 1]; w++)
        //            {
        //                _network.weights[l][n][w] += _network.weightDiff[l][n][w] + momentum * _network.prevWeightDiff[l][n][w];
        //                _network.prevWeightDiff[l][n][w] = _network.weightDiff[l][n][w];
        //            }
        //        }
        //    }
        //}
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
        public static double TransferFunction(MLPNew network, double x)
        {
            double result = 0;
            if (network.baseData._isSigmoidFunction)
                result = SigmoidTransferFunction(x);
            else
                result = HyperbolicTransferFunction(x);
            return result;
        }
        public double DerivativeFunction(double x)
        {
            return _network.baseData._isSigmoidFunction ? SigmoidDerivative(x) : HyperbolicDerivative(x);
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

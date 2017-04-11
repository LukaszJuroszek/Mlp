using Alea;
using Alea.CSharp;
using MLPProgram.Networks;
using System;
using System.Linq;

namespace MLPProgram.LearningAlgorithms
{
    public struct GradientLearning
    {

        [GpuParam] public double _etaPlus, _etaMinus, _minDelta, _maxDelta, _errorExponent;
        [GpuParam] public MLP _network;
        [GpuParam] public BaseDataHolder _baseData;
        public GradientLearning(MLP network, BaseDataHolder baseData)
        {
            _etaPlus = 1.2;
            _etaMinus = 0.5;
            _minDelta = 0.00001;
            _maxDelta = 10;
            _errorExponent = 2.0;
            _network = network;
            _baseData = baseData;
        }
        public void TrainOld(int numberOfEpochs = 30, int batchSize = 30, double learnRate = 0.05, double momentum = 0.5)
        {
            batchSize = _baseData._numberOFVectors;
            var gpu = Gpu.Default;
            var lp = new LaunchParam(16, 256);
            //var network = gpu.Allocate<MLP>(this);
            CreateWeightZeroAndAsingDeltaValue(0.1);
            for (var epoch = 0; epoch < numberOfEpochs; epoch++)
            {
                MakeGradientZero();
                for (var row = 0; row < batchSize; row++)
                {
                    var curentRowToOutput = _baseData._trainingDataSet[row].Take(_network.output[0].Length).ToArray();
                    ForwardPass(_network, _baseData, curentRowToOutput);
                    for (var l = 0; l < _baseData._numberOfOutput; l++)
                        _network.signalError[_network.numberOfLayer - 1][l] = CalculateSignalErrors(row, l);
                    for (var l = _network.numberOfLayer - 2; l > 0; l--)
                        for (var n = 0; n < _network.layer[l]; n++)
                            _network.signalError[l][n] = CalculateDerivativeForHiddenLayer(l, n) * SumSignalErrorForHiddenLayer(l, n);
                    for (var l = _network.numberOfLayer - 1; l > 0; l--)
                        for (var n = 0; n < _network.layer[l]; n++)
                            Bias(learnRate, l, n);
                }
                UpdateWeightsRprop(learnRate, momentum, _etaPlus, _etaMinus, _minDelta, _maxDelta);
                // zero-out gradients
                MakeGradientZero();
            }
        }
        public void Train(int numberOfEpochs = 30, int batchSize = 30, double learnRate = 0.05, double momentum = 0.5)
        {
            batchSize = _baseData._numberOFVectors;
            var gpu = Gpu.Default;
            var lp = new LaunchParam(16, 256);
            //var network = gpu.Allocate<MLP>(this);
            CreateWeightZeroAndAsingDeltaValue(0.1);
            for (var epoch = 0; epoch < numberOfEpochs; epoch++)
            {
                MakeGradientZero();
                for (var row = 0; row < batchSize; row++)
                {
                    var curentRowToOutput = _baseData._trainingDataSet[row].Take(_network.output[0].Length).ToArray();
                    ForwardPass(_network, _baseData, curentRowToOutput);
                    for (var l = 0; l < _baseData._numberOfOutput; l++)
                    {
                        _network.signalError[_network.numberOfLayer - 1][l] = CalculateSignalErrorsForOutputLayer(
                            _baseData._trainingDataSet[row][_baseData._numberOfInput + l] - _network.output[_network.numberOfLayer - 1][l], _network.output[_network.numberOfLayer - 1][l], _errorExponent, _network.classification);
                    }
                    for (var l = _network.numberOfLayer - 2; l > 0; l--)
                    {
                        for (var n = 0; n < _network.layer[l]; n++)
                        {
                            _network.signalError[l][n] = DerivativeFunction(_network.output[l][n], _network.classification)
                                * SumSignalErrorForHiddenLayer(_network.layer[l + 1], _network.signalError[l + 1], _network.weights[l + 1], n);
                        }
                    }
                    for (var l = _network.numberOfLayer - 1; l > 0; l--)
                    {
                        for (var n = 0; n < _network.layer[l]; n++)
                        {
                            _network.weightDiff[l][n][_network.layer[l - 1]] += learnRate * _network.signalError[l][n];
                            for (var w = 0; w < _network.layer[l - 1]; w++)
                                _network.weightDiff[l][n][w] += learnRate * _network.signalError[l][n] * _network.output[l - 1][w];
                        } 
                    }
                }
                UpdateWeightsRprop(learnRate, momentum, _etaPlus, _etaMinus, _minDelta, _maxDelta);
                // zero-out gradients
                MakeGradientZero();
            }
        }
        public static void ForwardPass(MLP network, BaseDataHolder baseData, double[] row, int lok = -1)
        {
            network.output[0] = row;
            for (var l = 1; l < network.output.Length; l++)
            {
                for (var n = 0; n < network.output[l].Length; n++)
                {
                    double sum = 0;
                    for (var w = 0; w < network.output[l - 1].Length; w++)
                    {
                        sum += network.output[l - 1][w] * network.weights[l][n][w];
                    }
                    sum += network.weights[l][n][network.output[l - 1].Length]; //bias
                    network.output[l][n] = (l == network.output.Length - 1 && !network.classification) ? sum : TransferFunction(baseData, sum);
                }
            }
        }
        public static int Sign(double number)
        {
            return number > 0 ? 1 : number < 0 ? -1 : 0;
        }
        private double CalculateSignalErrors(int v, int n)
        {
            var error = _baseData._trainingDataSet[v][_baseData._numberOfInput + n] - _network.output[_network.numberOfLayer - 1][n];
            error = Math.Sign(error) * Math.Pow(Math.Abs(error), _errorExponent);
            var derivative = CalculateDerivativeForSignalErrorsInOutputLayer(n);
            return error * derivative;
        }
        private double CalculateSignalErrorsForOutputLayer(double error, double outputValue, double errorExponent, bool classification)
        {
            error = Math.Sign(error) * Math.Pow(Math.Abs(error), errorExponent);
            var derivative = CalculateDerivativeForSignalErrors(classification, outputValue);
            return error * derivative;
        }
        private void Bias(double learnRate, int l, int n)
        {
            _network.weightDiff[l][n][_network.layer[l - 1]] += learnRate * _network.signalError[l][n];
            for (var w = 0; w < _network.layer[l - 1]; w++)
                _network.weightDiff[l][n][w] += learnRate * _network.signalError[l][n] * _network.output[l - 1][w];
        }
        private double SumSignalErrorForHiddenLayer(int layer, int hiddenLayerSecondDim)
        {
            var sum = 0.0;
            for (var w = 0; w < _network.layer[layer + 1]; w++)
                sum += _network.signalError[layer + 1][w] * _network.weights[layer + 1][w][hiddenLayerSecondDim];
            return sum;
        }
        private double SumSignalErrorForHiddenLayer(int inLayerValue, double[] signalError, double[][] weights, int n)
        {
            var sum = 0.0;
            for (var w = 0; w < inLayerValue; w++)
                sum += signalError[w] * weights[w][n];
            return sum;
        }
        private double CalculateDerivativeForHiddenLayer(int layer, int hidenLayeerSecondDim)
        {
            return DerivativeFunction(_network.output[layer][hidenLayeerSecondDim]);
        }
        private double CalculateDerivativeForSignalErrorsInOutputLayer(int l)
        {
            double derivative;
            if (_network.classification)
                derivative = DerivativeFunction(_network.output[_network.numberOfLayer - 1][l]);
            else
                derivative = 1.0;
            return derivative;
        }
        private double CalculateDerivativeForSignalErrors(bool classification, double x)
        {
            return classification ? DerivativeFunction(x) : 1.0;
        }
        private void CreateWeightZeroAndAsingDeltaValue(double deltaValue)
        {
            for (var l = 1; l < _network.numberOfLayer; l++)
                for (var n = 0; n < _network.layer[l]; n++)
                {
                    for (var w = 0; w <= _network.layer[l - 1]; w++)
                    {
                        _network.weightDiff[l][n][w] = 0;
                        _network.delta[l][n][w] = deltaValue;
                    }
                }
        }
        private void MakeGradientZero()
        {
            for (var l = 1; l < _network.numberOfLayer; l++)
                for (var n = 0; n < _network.layer[l]; n++)
                    for (var w = 0; w <= _network.layer[l - 1]; w++)
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
            for (var l = _network.numberOfLayer - 1; l > 0; l--)
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
            for (var l = _network.numberOfLayer - 1; l > 0; l--)
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
        public static double TransferFunction(BaseDataHolder baseData, double x)
        {
            double result = 0;
            if (baseData._isSigmoidFunction)
                result = SigmoidTransferFunction(x);
            else
                result = HyperbolicTransferFunction(x);
            return result;
        }
        public static double TransferFunction(bool isSigmoidFunction, double x)
        {
            double result = 0;
            if (isSigmoidFunction)
                result = SigmoidTransferFunction(x);
            else
                result = HyperbolicTransferFunction(x);
            return result;
        }
        public double DerivativeFunction(double x)
        {
            double result = 0;
            if (_baseData._isSigmoidFunction)
                result = SigmoidDerivative(x);
            else
                result = HyperbolicDerivative(x);
            return result;
        }
        public double DerivativeFunction(double x, bool isSigmoidFunction)
        {
            double result = 0;
            if (isSigmoidFunction)
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

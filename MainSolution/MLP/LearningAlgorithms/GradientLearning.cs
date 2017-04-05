using Alea;
using Alea.CSharp;
using Alea.Parallel;
using MLPProgram.Networks;
using System;
using System.Linq;

namespace MLPProgram.LearningAlgorithms
{
    public struct GradientLearning
    {
        [GpuParam]
        public double _etaPlus, _etaMinus, _minDelta, _maxDelta, _errorExponent;
        [GpuParam]
        public MLP _network;
        public GradientLearning(MLP network)
        {
            _etaPlus = 1.2;
            _etaMinus = 0.5;
            _minDelta = 0.00001;
            _maxDelta = 10;
            _errorExponent = 2.0;
            _network = network;
            Settings.Instance.Memory.AllowNonBlittableMemoryTransfer = true;
        }
        [GpuManaged]
        public MLP Train(int numberOfEpochs = 30, int batchSize = 30, double learnRate = 0.05, double momentum = 0.5)
        {
            //if (batchSize > _network.baseData._numberOFVectors || nameof(UpdateWeightsRprop).Contains("Rprop"))
            batchSize = _network.baseData._numberOFVectors;
            var gpu = Gpu.Default;
            var lp = new LaunchParam(16, 256);
            var trainingDataSet = _network.baseData._trainingDataSet;
            var isSigmoidFunction = _network.baseData._isSigmoidFunction;
            var numberOfOutput = _network.baseData._numberOfOutput;
            var numberOfInput = _network.baseData._numberOfInput;
            var output = _network.output;
            var weights = _network.weights;
            var classification = _network.classification;
            var signalError = _network.signalError;
            var weightDiff = _network.weightDiff;
            var numberOFLayers = _network.numbersOfLayers;
            var prevWeightDiff = _network.prevWeightDiff;
            var delta = _network.delta;
            var layer = _network.layer;
            var errorExponent = _errorExponent;
            CreateWeightZeroAndAsingDeltaValue(numberOFLayers, layer, weightDiff, delta, 0.1);
            for (var epoch = 0; epoch < numberOfEpochs; epoch++)
            {
                MakeGradientZero(numberOFLayers, layer, weightDiff);
                for (var v = 0; v < batchSize; v++)
                {
                    //gpu.For(0, batchSize, v =>
                    //{
                    Program.ForwardPass(output, trainingDataSet, weights, classification, isSigmoidFunction, v);
                    for (var l = 0; l < numberOfOutput; l++)
                    {
                        var error = trainingDataSet[v][numberOfInput + l] - output[numberOFLayers - 1][l];
                        error = Sign(error) * DeviceFunction.Pow(DeviceFunction.Abs(error), errorExponent);
                        double derivative;
                        if (classification)
                            derivative = DerivativeFunction(isSigmoidFunction, output[numberOFLayers - 1][l]);
                        else
                            derivative = 1.0;
                        signalError[numberOFLayers - 1][l] = error * derivative;
                    }

                    for (var l = numberOFLayers - 2; l > 0; l--)
                        for (var n = 0; n < layer[l]; n++)
                        {
                            var sum = 0.0;
                            for (var w = 0; w < layer[l + 1]; w++)
                                sum += signalError[l + 1][w] * weights[l + 1][w][n];
                            signalError[l][n] = DerivativeFunction(isSigmoidFunction, output[l][n]) * sum;
                        }

                    for (var l = numberOFLayers - 1; l > 0; l--)
                    {
                        for (var n = 0; n < layer[l]; n++)
                        {

                            weightDiff[l][n][layer[l - 1]] += learnRate * signalError[l][n];
                            for (var w = 0; w < layer[l - 1]; w++)
                                weightDiff[l][n][w] += learnRate * signalError[l][n] * output[l - 1][w];
                        }
                    }
                }
                //});
                UpdateWeightsRprop(layer, numberOFLayers, weights, weightDiff, prevWeightDiff, delta, learnRate, momentum, _etaPlus, _etaMinus, _minDelta, _maxDelta);
                // zero-out gradients
                MakeGradientZero(numberOFLayers, layer, weightDiff);
            }
            return new MLP
            {
                baseData = new BaseDataHolder
                {
                    _trainingDataSet = trainingDataSet,
                    _isSigmoidFunction = isSigmoidFunction,
                    _numberOfOutput = numberOfOutput,
                    _numberOfInput = numberOfInput
                },
                output = output,
                weights = weights,
                classification = classification,
                signalError = signalError,
                weightDiff = weightDiff,
                numbersOfLayers = numberOFLayers,
                prevWeightDiff = prevWeightDiff,
                delta = delta,
                layer = layer,
            };
        }
        public static int Sign(double number)
        {
            return number > 0 ? 1 : number < 0 ? -1 : 0;
        }

        private double CalculateSignalErrors(int v, int n)
        {
            var error = _network.baseData._trainingDataSet[v][_network.baseData._numberOfInput + n] - _network.output[_network.numbersOfLayers - 1][n];
            error = Math.Sign(error) * Math.Pow(Math.Abs(error), _errorExponent);
            var derivative = CalculateDerivativeForSignalErrorsInOutputLayer(n);
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
            for (var l = 1; l < _network.numbersOfLayers; l++)
                for (var n = 0; n < _network.layer[l]; n++)
                {
                    for (var w = 0; w <= _network.layer[l - 1]; w++)
                    {
                        _network.weightDiff[l][n][w] = 0;
                        _network.delta[l][n][w] = deltaValue;
                    }
                }
        }
        private static void CreateWeightZeroAndAsingDeltaValue(int numberOFLayers, int[] layer, double[][][] weightDiff, double[][][] delta, double deltaValue)
        {
            for (var l = 1; l < numberOFLayers; l++)
                for (var n = 0; n < layer[l]; n++)
                {
                    for (var w = 0; w <= layer[l - 1]; w++)
                    {
                        weightDiff[l][n][w] = 0;
                        delta[l][n][w] = deltaValue;
                    }
                }
        }
        private void MakeGradientZero()
        {
            for (var l = 1; l < _network.numbersOfLayers; l++)
                for (var n = 0; n < _network.layer[l]; n++)
                    for (var w = 0; w <= _network.layer[l - 1]; w++)
                        _network.weightDiff[l][n][w] = 0;
        }
        private static void MakeGradientZero(int numberOFLayers, int[] layer, double[][][] weightDiff)
        {
            for (var l = 1; l < numberOFLayers; l++)
                for (var n = 0; n < layer[l]; n++)
                    for (var w = 0; w <= layer[l - 1]; w++)
                        weightDiff[l][n][w] = 0;
        }
        public static void Kernel(Func<int, double> op, double[] result, int[] arg1)
        {
            var start = blockIdx.x * blockDim.x + threadIdx.x;
            var stride = gridDim.x * blockDim.x;
            for (var i = start; i < result.Length; i += stride)
            {
                result[i] = op(arg1[i]);
            }
        }
        public double Test(double[][] trainingDataSet, double[][] testDataSet)
        {
            return _network.Accuracy(out var errorsRMSE, 0);
        }
        public static void UpdateWeightsRprop(
            int[] layer,
            int numberOfLayers,
            double[][][] weights,
            double[][][] weightDiff,
            double[][][] prevWeightDiff,
            double[][][] delta,
           double learnRate,
           double momentum,
           double etaPlus,
           double etaMinus,
           double minDelta,
           double maxDelta,
           double inputWeightRegularizationCoef = -1)
        {
            for (var l = numberOfLayers - 1; l > 0; l--)
            {
                for (var n = 0; n < layer[l]; n++)
                {
                    for (var w = 0; w <= layer[l - 1]; w++)
                    {
                        if (inputWeightRegularizationCoef <= 0 || weights[l][n][w] != 0)
                        {
                            if (prevWeightDiff[l][n][w] * weightDiff[l][n][w] > 0)
                            {
                                delta[l][n][w] *= etaPlus;
                                if (delta[l][n][w] > maxDelta)
                                    delta[l][n][w] = maxDelta;
                            }
                            else if (prevWeightDiff[l][n][w] * weightDiff[l][n][w] < 0)
                            {
                                delta[l][n][w] *= etaMinus;
                                if (delta[l][n][w] < minDelta)
                                    delta[l][n][w] = minDelta;
                            }
                            weights[l][n][w] += Sign(weightDiff[l][n][w]) * delta[l][n][w];
                            prevWeightDiff[l][n][w] = weightDiff[l][n][w];
                        }
                        else
                        {
                            prevWeightDiff[l][n][w] = 0;
                            weightDiff[l][n][w] = 0;
                        }
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
            for (var l = _network.numbersOfLayers - 1; l > 0; l--)
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
        public static double TransferFunction(MLP _network, double x)
        {
            double result = 0;
            if (_network.baseData._isSigmoidFunction)
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
            if (_network.baseData._isSigmoidFunction)
                result = SigmoidDerivative(x);
            else
                result = HyperbolicDerivative(x);
            return result;
        }
        public static double DerivativeFunction(bool isSigmoidFunction, double x)
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

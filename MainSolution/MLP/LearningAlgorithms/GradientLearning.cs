using Alea;
using Alea.CSharp;
using MLPProgram.Networks;
using System;
using System.Linq;

namespace MLPProgram.LearningAlgorithms
{
    public struct GradientLearning
    {
        [GpuParam]
        public double _etaPlus, _etaMinus, _minDelta , _maxDelta, _errorExponent;
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
        }
        [GpuManaged]
        public void Train(MLP _network,int numberOfEpochs = 30, int batchSize = 30, double learnRate = 0.05, double momentum = 0.5)
        {
            if (batchSize > _network.baseData._numberOFVectors || nameof(UpdateWeightsRprop).Contains("Rprop"))
                batchSize = _network.baseData._numberOFVectors;
            var _gpu = Gpu.Default;
            var lp = new LaunchParam(16, 256);
            var op = _network.output;
            Func <int, double> calculateDerivative = x => op[2][x] * (1.0 - op[2][x]);
            Func<double, double, double> calculateError = (x, y) => Sign(x - y) * DeviceFunction.Pow(DeviceFunction.Abs(x - y), 2.0);
            Func<double, double, double> calculateErrorSignal = (x, y) => x * y;
            CreateWeightZeroAndAsingDeltaValue(0.1);
            Console.WriteLine(numberOfEpochs* _network.baseData._numberOFVectors*batchSize);
            for (var epoch = 0; epoch < numberOfEpochs; epoch++)
            {
                MakeGradientZero();
                var v = 0;// v ?
                while (v < _network.baseData._numberOFVectors)
                {
                    for (var b = 0; b < batchSize; b++)
                    {
                        Program.ForwardPass(_network, v);
                        double[] derivatiesTmp = Enumerable.Repeat(0.0, _network.baseData._numberOfOutput).ToArray();
                        double[] errorsTmp = Enumerable.Repeat(0.0, _network.baseData._numberOfOutput).ToArray();
                        var derivatiesTmprange = Enumerable.Range(0, _network.baseData._numberOfOutput).ToArray();
                        //var derivatiesRange = Enumerable.Range(0, _network.baseData._numberOfOutput).ToArray();
                        var se = _network.signalError[_network.numLayers - 1];
                        var tds = _network.baseData._trainingDataSet[v];
                        var ou = _network.output[_network.numLayers - 1];
                        //_network.baseData._trainingDataSet[v][_network.baseData._numberOfInput + n] - _network.output[_network.numLayers - 1][n]
                        var erro1 = _network.baseData._trainingDataSet[v];
                        var error2= _network.output[_network.numLayers - 1];
                       var numberOfInp = _network.baseData._numberOfInput;
                        _gpu.Launch(Kernel, lp, calculateDerivative, derivatiesTmp, derivatiesTmprange);
                        _gpu.Launch(Kernel, lp, calculateError, errorsTmp, erro1,error2,numberOfInp);
                        //Console.WriteLine($"{errorsTmp.Count()==derivatiesTmp.Count()} {iteratino++}" );
                        _gpu.Launch(Kernel, lp, calculateErrorSignal, se, errorsTmp,derivatiesTmp);
                        _network.signalError[_network.numLayers - 1] = se;
                        //for (var n = 0; n < _network.baseData._numberOfOutput; n++)
                        /*_network.signalError[_network.numLayers - 1][n] = CalculateSignalErrors(v, n);*/
                        for (var l = _network.numLayers - 2; l > 0; l--)
                            for (var n = 0; n < _network.layer[l]; n++)
                                _network.signalError[l][n] = CalculateDerivativeForHiddenLayer(l, n) * SumSignalErrorForHiddenLayer(l, n);
                        for (var l = _network.numLayers - 1; l > 0; l--)
                            for (var n = 0; n < _network.layer[l]; n++)
                                Bias(learnRate, l, n);
                        v++;
                        if (v == _network.baseData._numberOFVectors)
                            break;
                    }
                    UpdateWeightsRprop(learnRate, momentum, _etaPlus, _etaMinus, _minDelta, _maxDelta);
                    // zero-out gradients
                    MakeGradientZero();
                }
            }
        }
        public static int Sign(double number)
        {
            return number > 0 ? 1 : number < 0 ? -1 : 0;
        }
        private double CalculateSignalErrors(int v, int n)
        {
            var error = _network.baseData._trainingDataSet[v][_network.baseData._numberOfInput + n] - _network.output[_network.numLayers - 1][n];
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
                derivative = DerivativeFunction(_network.output[_network.numLayers - 1][outputSecondDim]);
            else
                derivative = 1.0;
            return derivative;
        }
        private void CreateWeightZeroAndAsingDeltaValue(double deltaValue)
        {
            Func<double, double, double> assing = (x, y) => { return y; };
            var delta = Enumerable.Repeat(deltaValue, _network.delta[1][1].Length);
            var wz = Enumerable.Repeat(0.0, _network.weightDiff[1][1].Length);
            for (var l = 1; l < _network.numLayers; l++)
                for (var n = 0; n < _network.layer[l]; n++)
                {
                    for (var w = 0; w <= _network.layer[l - 1]; w++)
                    {
                        _network.weightDiff[l][n][w] = 0;
                        _network.delta[l][n][w] = deltaValue;
                    }
                    //_gpu.Launch(Kernel, new LaunchParam(16, 256), assing, _network.delta[l][n], delta.ToArray());
                    //_gpu.Launch(Kernel, new LaunchParam(16, 256), assing, _network.weightDiff[l][n], wz.ToArray());
                }
        }
        private void MakeGradientZero()
        {
            for (var l = 1; l < _network.numLayers; l++)
                for (var n = 0; n < _network.layer[l]; n++)
                    for (var w = 0; w <= _network.layer[l - 1]; w++)
                        _network.weightDiff[l][n][w] = 0;
        }
        public static void Kernel<T>(Func<T, T, T> op, T[] result, T[] arg1,T[]arg2,int numOfinputs)
        {
            var start = blockIdx.x * blockDim.x + threadIdx.x;
            var stride = gridDim.x * blockDim.x;
            for (var i = start; i < result.Length; i += stride)
            {
                result[i] = op(arg1[numOfinputs+i],arg2[i]);
            }
        }
        public static void Kernel<T>(Func<T, T, T> op, T[] result, T[] arg1, T[] arg2)
        {
            var start = blockIdx.x * blockDim.x + threadIdx.x;
            var stride = gridDim.x * blockDim.x;
            for (var i = start; i < result.Length; i += stride)
            {
                result[i] = op(arg1[i], arg2[i]);
            }
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

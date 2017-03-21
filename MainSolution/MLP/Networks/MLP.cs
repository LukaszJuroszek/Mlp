using Alea;
using MLPProgram.TransferFunctions;
using System;
namespace MLPProgram.Networks
{
    class MLP : INetwork
    {
        [GpuParam]
        public double[] _featureImportance;
        [GpuParam]
        public int[] _featureNumber;
        [GpuParam]
        private Random _rnd;
        [GpuParam]
        public double[][][] _weightDiff;
        [GpuParam]
        public double[][][] _prevWeightDiff;
        [GpuParam]
        public double[][][] _delta;
        [GpuParam]
        public double[][][] _weights;
        [GpuParam]
        public double[][] _signalError;
        [GpuParam]
        public double[][] _output;
        [GpuParam]
        public int[] _layer;
        [GpuParam]
        public int _numLayers;
        [GpuParam]
        public ITransferFunction _transferFunction;
        [GpuParam]
        public bool _classification;
        [GpuParam]
        public int _numWeights;
        public MLP(int[] layer, bool classification, ITransferFunction transferFunction, string weightFile = "")
        {
            InitFilds(layer, classification, transferFunction);
            var dw0 = 0.20;
            for (var l = 1; l < _numLayers; l++)
                for (var n = 0; n < layer[l]; n++)
                    for (var w = 0; w < layer[l - 1] + 1; w++)
                    {
                        _weights[l][n][w] = 0.4 * (0.5 - _rnd.NextDouble());
                        _delta[l][n][w] = dw0; //for VSS and Rprop
                    }
        }
        private void InitFilds(int[] layer, bool classification, ITransferFunction transferFunction)
        {
            _classification = classification;
            _layer = layer;
            _transferFunction = transferFunction;
            _numLayers = layer.Length;
            _weights = new double[_numLayers][][];
            _weightDiff = new double[_numLayers][][];
            _delta = new double[_numLayers][][];
            _signalError = new double[_numLayers][];
            _output = new double[_numLayers][];
            _output[0] = new double[layer[0]];
            _numWeights = 0;
            _prevWeightDiff = new double[_numLayers][][];
            _rnd = new Random();
            for (var l = 1; l < _numLayers; l++)
            {
                InitSecondDimension(layer, l);
                for (var n = 0; n < layer[l]; n++)
                {
                    InitTrirdDimension(layer, l, n);
                    _numWeights++;
                }
            }
        }
        private void InitSecondDimension(int[] layer, int l)
        {
            _weights[l] = new double[layer[l]][];
            _weightDiff[l] = new double[layer[l]][];
            _prevWeightDiff[l] = new double[layer[l]][];
            _delta[l] = new double[layer[l]][];
            _signalError[l] = new double[layer[l]];
            _output[l] = new double[layer[l]];
        }
        private void InitTrirdDimension(int[] layer, int l, int n)
        {
            _weights[l][n] = new double[layer[l - 1] + 1];
            _weightDiff[l][n] = new double[layer[l - 1] + 1];
            _prevWeightDiff[l][n] = new double[layer[l - 1] + 1];
            _delta[l][n] = new double[layer[l - 1] + 1];
        }
        public double Accuracy(double[][] dataSet, out double error, ITransferFunction transferFunction, int lok = 0)
        {
            double maxValue = -1;
            error = 0.0;
            var classification = false;
            if (dataSet[0].Length > _layer[0] + 1)
                classification = true;
            var numCorrect = 0;
            var maxIndex = -1;
            for (var v = 0; v < dataSet.Length; v++)
            {
                ForwardPass(dataSet[v], transferFunction, lok);
                maxIndex = -1;
                maxValue = -1.1;
                for (var n = 0; n < _layer[_numLayers - 1]; n++)
                {
                    if (classification)
                        error += transferFunction.TransferFunction(_output[_numLayers - 1][n] - (2 * dataSet[v][_layer[0] + n] - 1));
                    else
                        error += Math.Pow(_output[_numLayers - 1][n] - dataSet[v][_layer[0] + n], 2);
                    if (_output[_numLayers - 1][n] > maxValue)
                    {
                        maxValue = _output[_numLayers - 1][n];
                        maxIndex = n;
                    }
                }
                var position = _layer[0] + maxIndex;
                if (dataSet[v][position] == 1)
                    numCorrect++;
            }
            error /= dataSet.Length;
            return (double)numCorrect / dataSet.Length;
        }
        public void ForwardPass(double[] vector, ITransferFunction transferFunction, int lok = -1)
        {
            for (var i = 0; i < _layer[0]; i++)
                _output[0][i] = vector[i];
            for (var l = 1; l < _numLayers; l++)
            {
                for (var n = 0; n < _layer[l]; n++)
                {
                    double sum = 0;
                    for (var w = 0; w < _layer[l - 1]; w++)
                    {
                        sum += _output[l - 1][w] * _weights[l][n][w];
                    }
                    sum += _weights[l][n][_layer[l - 1]]; //bias
                    if (l == _numLayers - 1 && !_classification)
                        _output[l][n] = sum;
                    else
                        _output[l][n] = transferFunction.TransferFunction(sum);
                }
            }
        }
        public double[] GetNonSignalErrorTable(double[][] DataSet, ref double accuracy, double errorExponent = 2.0)
        {
            var numVect = DataSet.Length;
            double[] errorTable = new double[numVect];
            double error = 0;
            for (var v = 0; v < numVect; v++)
            {
                error = 0;
                for (var n = 0; n < _layer[0]; n++)
                    _output[0][n] = DataSet[v][n];
                for (var l = 1; l < _numLayers; l++)
                {
                    for (var n = 0; n < _layer[l]; n++)
                    {
                        double sum = 0;
                        for (var w = 0; w < _layer[l - 1]; w++)
                            sum += _output[l - 1][w] * _weights[l][n][w];
                        sum += _weights[l][n][_layer[l - 1]];
                        if (l == _numLayers - 1 && !_classification)
                            _output[l][n] = sum;
                        else
                            _output[l][n] = _transferFunction.TransferFunction(sum);
                        if (l == _numLayers - 1)
                            error += Math.Pow(Math.Abs(_output[l][n] - DataSet[v][_layer[0] + n]), errorExponent);
                    }
                }
                errorTable[v] = error;
            }
            return errorTable;
        }
    }
}

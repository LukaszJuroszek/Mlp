using MLPProgram.TransferFunctions;
using System;
namespace MLPProgram.Networks
{
    class MLP : INetwork
    {
        public double[] _featureImportance;
        public int[] _featureNumber;
        private Random _rnd;
        public double[][][] WeightDiff { get; set; }
        public double[][][] Delta { get; set; }
        public double[][][] Weights { get; set; }
        public double[][] SignalError { get; set; }
        public double[][] Output { get; set; }
        public int[] Layer { get; set; }
        public int NumLayers { get; set; }
        public ITransferFunction TransferFunction { get; set; }
        public bool Classification { get; set; }
        public int NumWeights { get; set; }
        public double[][][] PrevWeightDiff { get; set; }
        public MLP(int[] layer,bool classification,ITransferFunction transferFunction,string weightFile = "")
        {
            InitFilds(layer,classification,transferFunction);
            var dw0 = 0.20;
            for (var l = 1;l < NumLayers;l++)
                for (var n = 0;n < layer[l];n++)
                    for (var w = 0;w < layer[l - 1] + 1;w++)
                    {
                        Weights[l][n][w] = 0.4 * ( 0.5 - _rnd.NextDouble() );
                        Delta[l][n][w] = dw0; //for VSS and Rprop
                    }
        }
        private void InitFilds(int[] layer,bool classification,ITransferFunction transferFunction)
        {
            Classification = classification;
            Layer = layer;
            TransferFunction = transferFunction;
            NumLayers = layer.Length;
            Weights = new double[NumLayers][][];
            WeightDiff = new double[NumLayers][][];
            Delta = new double[NumLayers][][];
            SignalError = new double[NumLayers][];
            Output = new double[NumLayers][];
            Output[0] = new double[layer[0]];
            NumWeights = 0;
            PrevWeightDiff = new double[NumLayers][][];
            _rnd = new Random();
            for (var l = 1;l < NumLayers;l++)
            {
                InitSecondDimension(layer,l);
                for (var n = 0;n < layer[l];n++)
                {
                    InitTrirdDimension(layer,l,n);
                    NumWeights++;
                }
            }
        }
        private void InitSecondDimension(int[] layer,int l)
        {
            Weights[l] = new double[layer[l]][];
            WeightDiff[l] = new double[layer[l]][];
            PrevWeightDiff[l] = new double[layer[l]][];
            Delta[l] = new double[layer[l]][];
            SignalError[l] = new double[layer[l]];
            Output[l] = new double[layer[l]];
        }
        private void InitTrirdDimension(int[] layer,int l,int n)
        {
            Weights[l][n] = new double[layer[l - 1] + 1];
            WeightDiff[l][n] = new double[layer[l - 1] + 1];
            PrevWeightDiff[l][n] = new double[layer[l - 1] + 1];
            Delta[l][n] = new double[layer[l - 1] + 1];
        }
        public double Accuracy(double[][] dataSet,out double error,ITransferFunction transferFunction,int lok = 0)
        {
            double maxValue = -1;
            error = 0.0;
            var classification = false;
            if (dataSet[0].Length > Layer[0] + 1)
                classification = true;
            var numCorrect = 0;
            var maxIndex = -1;
            for (var v = 0;v < dataSet.Length;v++)
            {
                ForwardPass(dataSet[v],transferFunction,lok);
                maxIndex = -1;
                maxValue = -1.1;
                for (var n = 0;n < Layer[NumLayers - 1];n++)
                {
                    if (classification)
                        error += transferFunction.TransferFunction(Output[NumLayers - 1][n] - ( 2 * dataSet[v][Layer[0] + n] - 1 ));
                    else
                        error += Math.Pow(Output[NumLayers - 1][n] - dataSet[v][Layer[0] + n],2);
                    if (Output[NumLayers - 1][n] > maxValue)
                    {
                        maxValue = Output[NumLayers - 1][n];
                        maxIndex = n;
                    }
                }
                var position = Layer[0] + maxIndex;
                if (dataSet[v][position] == 1)
                    numCorrect++;
            }
            error /= dataSet.Length;
            return (double)numCorrect / dataSet.Length;
        }
        public void ForwardPass(double[] vector,ITransferFunction transferFunction,int lok = -1)
        {
            for (var i = 0;i < Layer[0];i++)
                Output[0][i] = vector[i];
            for (var l = 1;l < NumLayers;l++)
            {
                for (var n = 0;n < Layer[l];n++)
                {
                    double sum = 0;
                    for (var w = 0;w < Layer[l - 1];w++)
                    {
                        sum += Output[l - 1][w] * Weights[l][n][w];
                    }
                    sum += Weights[l][n][Layer[l - 1]]; //bias
                    if (l == NumLayers - 1 && !Classification)
                        Output[l][n] = sum;
                    else
                        Output[l][n] = transferFunction.TransferFunction(sum);
                }
            }
        }
        public double[] GetNonSignalErrorTable(double[][] DataSet,ref double accuracy,double errorExponent = 2.0)
        {
            var numVect = DataSet.Length;
            double[] errorTable = new double[numVect];
            double error = 0;
            for (var v = 0;v < numVect;v++)
            {
                error = 0;
                for (var n = 0;n < Layer[0];n++)
                    Output[0][n] = DataSet[v][n];
                for (var l = 1;l < NumLayers;l++)
                {
                    for (var n = 0;n < Layer[l];n++)
                    {
                        double sum = 0;
                        for (var w = 0;w < Layer[l - 1];w++)
                            sum += Output[l - 1][w] * Weights[l][n][w];
                        sum += Weights[l][n][Layer[l - 1]];
                        if (l == NumLayers - 1 && !Classification)
                            Output[l][n] = sum;
                        else
                            Output[l][n] = TransferFunction.TransferFunction(sum);
                        if (l == NumLayers - 1)
                            error += Math.Pow(Math.Abs(Output[l][n] - DataSet[v][Layer[0] + n]),errorExponent);
                    }
                }
                errorTable[v] = error;
            }
            return errorTable;
        }
    }
}

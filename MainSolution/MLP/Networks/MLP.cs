﻿using Alea;
using System;
namespace MLPProgram.Networks
{
    public struct MLP : INetwork
    {
        //public double[] featureImportance;
        //public int[] featureNumber;
        //[GpuParam]
        //public Random rnd;
        [GpuParam]
        public double[][][] weightDiff;
        [GpuParam]
        public double[][][] prevWeightDiff;
        [GpuParam]
        public double[][][] delta;
        [GpuParam]
        public double[][][] weights;
        [GpuParam]
        public double[][] signalError;
        [GpuParam]
        public double[][] output;
        [GpuParam]
        public int[] layer;
        [GpuParam]
        public int numLayers;
        [GpuParam]
        public Func<double, double> transferFunction;
        [GpuParam]
        public bool classification;
        [GpuParam]
        public int numWeights;
        public DataFileHolder dataFileHolder;
        public MLP(DataFileHolder fileHolder, string weightFile = "")
        {
            dataFileHolder = fileHolder;
            classification = dataFileHolder.Classification ;
            layer = dataFileHolder.GetLayers();
            transferFunction = dataFileHolder.TransferFunction ;
            numWeights = 0;
            numLayers = layer.Length;
            weights = new double[numLayers][][];
            weightDiff = new double[numLayers][][];
            delta = new double[numLayers][][];
            signalError = new double[numLayers][];
            output = new double[numLayers][];
            output[0] = new double[layer[0]];
            prevWeightDiff = new double[numLayers][][];
            var rnd = new Random();
            InitMultiDimArray(layer);
            var dw0 = 0.20;
            for (var l = 1; l < numLayers; l++)
                for (var n = 0; n < layer[l]; n++)
                    for (var w = 0; w < layer[l - 1] + 1; w++)
                    {
                        weights[l][n][w] = 0.4 * (0.5 - rnd.NextDouble());//create random weigths 
                        delta[l][n][w] = dw0; //for Rprop
                    }
        }
        private void InitMultiDimArray(int[] layer)
        {
            for (var l = 1; l < numLayers; l++)
            {
                weights[l] = new double[layer[l]][];
                weightDiff[l] = new double[layer[l]][];
                prevWeightDiff[l] = new double[layer[l]][];
                delta[l] = new double[layer[l]][];
                signalError[l] = new double[layer[l]];
                output[l] = new double[layer[l]];
                for (var n = 0; n < layer[l]; n++)
                {
                    weights[l][n] = new double[layer[l - 1] + 1];
                    weightDiff[l][n] = new double[layer[l - 1] + 1];
                    prevWeightDiff[l][n] = new double[layer[l - 1] + 1];
                    delta[l][n] = new double[layer[l - 1] + 1];
                    numWeights++;
                }
            }
        }
        public double Accuracy( out double error,  int lok = 0)
        {
            double maxValue = -1;
            error = 0.0;
            var classification = false;
            if (dataFileHolder.Data[0].Length > layer[0] + 1)
                classification = true;
            var numCorrect = 0;
            var maxIndex = -1;
            for (var v = 0; v < dataFileHolder.Data.Length; v++)
            {
                Program.ForwardPass(this,dataFileHolder.Data[v], transferFunction, lok);
                maxIndex = -1;
                maxValue = -1.1;
                for (var n = 0; n < layer[numLayers - 1]; n++)
                {
                    if (classification)
                        error += transferFunction(output[numLayers - 1][n] - (2 * dataFileHolder.Data[v][layer[0] + n] - 1));
                    else
                        error += Math.Pow(output[numLayers - 1][n] - dataFileHolder.Data[v][layer[0] + n], 2);
                    if (output[numLayers - 1][n] > maxValue)
                    {
                        maxValue = output[numLayers - 1][n];
                        maxIndex = n;
                    }
                }
                var position = layer[0] + maxIndex;
                if (dataFileHolder.Data[v][position] == 1)
                    numCorrect++;
            }
            error /= dataFileHolder.Data.Length;
            return (double)numCorrect / dataFileHolder.Data.Length;
        }

        public double[] GetNonSignalErrorTable(double[][] DataSet, ref double accuracy, double errorExponent = 2.0)
        {
            var numVect = DataSet.Length;
            double[] errorTable = new double[numVect];
            double error = 0;
            for (var v = 0; v < numVect; v++)
            {
                error = 0;
                for (var n = 0; n < layer[0]; n++)
                    output[0][n] = DataSet[v][n];
                for (var l = 1; l < numLayers; l++)
                {
                    for (var n = 0; n < layer[l]; n++)
                    {
                        double sum = 0;
                        for (var w = 0; w < layer[l - 1]; w++)
                            sum += output[l - 1][w] * weights[l][n][w];
                        sum += weights[l][n][layer[l - 1]];
                        if (l == numLayers - 1 && !classification)
                            output[l][n] = sum;
                        else
                            output[l][n] = transferFunction(sum);
                        if (l == numLayers - 1)
                            error += Math.Pow(Math.Abs(output[l][n] - DataSet[v][layer[0] + n]), errorExponent);
                    }
                }
                errorTable[v] = error;
            }
            return errorTable;
        }
    }
}

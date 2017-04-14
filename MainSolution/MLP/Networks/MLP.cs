﻿using Alea;
using Alea.Parallel;
using MLPProgram.LearningAlgorithms;
using System;
using System.Text;

namespace MLPProgram.Networks
{
    public struct MLP
    {
        public double[][][] weightDiff, prevWeightDiff, delta, weights;
        public double[][] signalError, output;
        public int[] layer;
        public int numbersOfLayers;
        public bool classification;
        public int numWeights;
        public BaseDataHolder baseData;
        public override string ToString()
        {
            var st = new StringBuilder();
            st.Append($"Weights {numWeights} ");
            st.Append($"classification {classification} ");
            st.Append($"LayersCount {numbersOfLayers} ");
            for (var i = 0; i < layer.Length; i++)
            {
                st.Append($"l[{i}]={layer[i]} ");
            }
            return st.ToString();
        }
        public MLP(BaseDataHolder data, string weightFile = "")
        {
            baseData = data;
            classification = data._classification;
            layer = data._layer;
            numWeights = 0;
            numbersOfLayers = layer.Length;
            weights = new double[numbersOfLayers][][];
            weightDiff = new double[numbersOfLayers][][];
            delta = new double[numbersOfLayers][][];
            signalError = new double[numbersOfLayers][];
            output = new double[numbersOfLayers][];
            output[0] = new double[layer[0]];
            prevWeightDiff = new double[numbersOfLayers][][];
            var rnd = new Random();
            for (int l = 1; l < numbersOfLayers; l++)
            {
                weights[l] = new double[layer[l]][];
                weightDiff[l] = new double[layer[l]][];
                prevWeightDiff[l] = new double[layer[l]][];
                delta[l] = new double[layer[l]][];
                signalError[l] = new double[layer[l]];
                output[l] = new double[layer[l]];
                for (int n = 0; n < layer[l]; n++)
                {
                    weights[l][n] = new double[layer[l - 1] + 1];
                    weightDiff[l][n] = new double[layer[l - 1] + 1];
                    prevWeightDiff[l][n] = new double[layer[l - 1] + 1];
                    delta[l][n] = new double[layer[l - 1] + 1];
                    numWeights++;
                }
            }
            double dw0 = 0.20;
            for (int l = 1; l < numbersOfLayers; l++)
                for (int n = 0; n < layer[l]; n++)
                    for (int w = 0; w < layer[l - 1] + 1; w++)
                    {
                        weights[l][n][w] = 0.4 * (0.5 - rnd.NextDouble());//create random weigths 
                        delta[l][n][w] = dw0; //for Rprop
                    }
        }
        public double Accuracy(int lok = 0)
        {
            double maxValue = -1;
            var error = 0.0;
            var classification = false;
            if (baseData._trainingDataSet[0].Length > layer[0] + 1)
                classification = true;
            var numCorrect = 0;
            var maxIndex = -1;
            for (var v = 0; v < baseData._trainingDataSet.Length; v++)
            {
                Program.ForwardPass(this, v, lok);
                maxIndex = -1;
                maxValue = -1.1;
                for (var n = 0; n < layer[numbersOfLayers - 1]; n++)
                {
                    if (classification)
                        error += GradientLearning.TransferFunction(this, output[numbersOfLayers - 1][n] - (2 * baseData._trainingDataSet[v][layer[0] + n] - 1));
                    else
                        error += Math.Pow(output[numbersOfLayers - 1][n] - baseData._trainingDataSet[v][layer[0] + n], 2);
                    if (output[numbersOfLayers - 1][n] > maxValue)
                    {
                        maxValue = output[numbersOfLayers - 1][n];
                        maxIndex = n;
                    }
                }
                var position = layer[0] + maxIndex;
                if (baseData._trainingDataSet[v][position] == 1)
                    numCorrect++;
            }
            error /= baseData._trainingDataSet.Length;
            Console.WriteLine(error);
            return (double)numCorrect / baseData._trainingDataSet.Length;
        }
        public static double Accuracy(MLP mlp, out double error, int lok = 0)
        {
            double maxValue = -1;
            error = 0.0;
            var classification = false;
            if (mlp.baseData._trainingDataSet[0].Length > mlp.layer[0] + 1)
                classification = true;
            var numCorrect = 0;
            var maxIndex = -1;
            for (var v = 0; v < mlp.baseData._trainingDataSet.Length; v++)
            {
                Program.ForwardPass(mlp, v, lok);
                maxIndex = -1;
                maxValue = -1.1;
                for (var n = 0; n < mlp.layer[mlp.numbersOfLayers - 1]; n++)
                {
                    if (classification)
                        error += GradientLearning.TransferFunction(mlp, mlp.output[mlp.numbersOfLayers - 1][n] - (2 * mlp.baseData._trainingDataSet[v][mlp.layer[0] + n] - 1));
                    else
                        error += Math.Pow(mlp.output[mlp.numbersOfLayers - 1][n] - mlp.baseData._trainingDataSet[v][mlp.layer[0] + n], 2);
                    if (mlp.output[mlp.numbersOfLayers - 1][n] > maxValue)
                    {
                        maxValue = mlp.output[mlp.numbersOfLayers - 1][n];
                        maxIndex = n;
                    }
                }
                var position = mlp.layer[0] + maxIndex;
                if (mlp.baseData._trainingDataSet[v][position] == 1)
                    numCorrect++;
            }
            error /= mlp.baseData._trainingDataSet.Length;
            return (double)numCorrect / mlp.baseData._trainingDataSet.Length;
        }
        public double Accuracy()
        {
            var network = this;
            var gpu = Gpu.Default;
            var isSigmoidFunction = network.baseData._isSigmoidFunction;
            var classification = network.classification;
            var numberOfLayers = network.numbersOfLayers;
            var trainingDataSet = gpu.Allocate(network.baseData._trainingDataSet);
            var trainingDataSetLenth = network.baseData._trainingDataSet.Length;
            var trainingDataSetLenthAt0 = network.baseData._trainingDataSet[0].Length;
            var output = gpu.Allocate(network.output);
            var weights = gpu.Allocate(network.weights);
            var layer = network.layer;

            //cout accuracy
            double maxValue = -1;
            var errorValue = new double[trainingDataSetLenth];
            if (trainingDataSetLenthAt0 > layer[0] + 1)
                classification = true;
            var numCorrect = new double[trainingDataSetLenth];
            var maxIndex = -1;
            try
            {
                gpu.For(0, trainingDataSetLenth, v =>
                      {
                          //    for (var v = 0; v < trainingDataSet.Length; v++)
                          //{
                          //Program.ForwardPass(this, v, 0);
                          for (var i = 0; i < output[0].Length; i++)
                              output[0][i] = trainingDataSet[v][i];
                          for (var l = 1; l < output.Length; l++)
                          {
                              for (var n = 0; n < output[l].Length; n++)
                              {
                                  double sum = 0;
                                  for (var w = 0; w < output[l - 1].Length; w++)
                                  {
                                      sum += output[l - 1][w] * weights[l][n][w];
                                  }
                                  sum += weights[l][n][output[l - 1].Length]; //bias
                                  output[l][n] = (l == output.Length - 1 && !classification) ? sum : GradientLearning.TransferFunction(isSigmoidFunction, sum);
                              }
                          }
                          maxIndex = -1;
                          maxValue = -1.1;
                          for (var n = 0; n < layer[numberOfLayers - 1]; n++)
                          {
                              if (classification)
                              {
                                  errorValue[v] += GradientLearning.TransferFunction(isSigmoidFunction, output[numberOfLayers - 1][n] - (2 * trainingDataSet[v][layer[0] + n] - 1));
                              }
                              else
                                  errorValue[v] += DeviceFunction.Pow(output[numberOfLayers - 1][n] - trainingDataSet[v][layer[0] + n], 2);
                              if (output[numberOfLayers - 1][n] > maxValue)
                              {
                                  maxValue = output[numberOfLayers - 1][n];
                                  maxIndex = n;
                              }
                          }
                          if (trainingDataSet[v][layer[0] + maxIndex] == 1)
                              numCorrect[v]++;
                          //}
                      });
                var sumError = gpu.Sum(errorValue);
                var sumCorrect = gpu.Sum(numCorrect);

                Console.WriteLine(sumCorrect);
                Console.WriteLine(sumCorrect / trainingDataSetLenth);
                return sumCorrect /= trainingDataSetLenth;
            }
            finally
            {
                Gpu.Free(trainingDataSet);
                Gpu.Free(output);
                Gpu.Free(weights);
            }
            throw new ArithmeticException("ZeroResult");
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
                for (var l = 1; l < numbersOfLayers; l++)
                {
                    for (var n = 0; n < layer[l]; n++)
                    {
                        double sum = 0;
                        for (var w = 0; w < layer[l - 1]; w++)
                            sum += output[l - 1][w] * weights[l][n][w];
                        sum += weights[l][n][layer[l - 1]];
                        if (l == numbersOfLayers - 1 && !classification)
                            output[l][n] = sum;
                        else
                            output[l][n] = GradientLearning.TransferFunction(this, sum);
                        if (l == numbersOfLayers - 1)
                            error += Math.Pow(Math.Abs(output[l][n] - DataSet[v][layer[0] + n]), errorExponent);
                    }
                }
                errorTable[v] = error;
            }
            return errorTable;
        }
    }
}

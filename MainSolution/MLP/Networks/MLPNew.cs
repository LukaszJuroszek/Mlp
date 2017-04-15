using MLPProgram.LearningAlgorithms;
using System;

namespace MLPProgram.Networks
{
    public enum NetworkLayer
    {
        Input = 0,
        Hidden = 1,
        Output = 2
    }
    public struct MLPNew
    {
        public double[][,] weightDiff, prevWeightDiff, delta, weights;
        public double[][] signalError, output;
        public int[] networkLayers;
        public int numbersOfLayers;
        public bool classification;
        public DataHolder baseData;
        public MLPNew(DataHolder data)
        {
            var rnd = new Random();
            baseData = data;
            classification = data._classification;
            networkLayers = data._layer;
            numbersOfLayers = networkLayers.Length;
            weights = Create2DLayers(networkLayers);
            weightDiff = Create2DLayers(networkLayers);
            prevWeightDiff = Create2DLayers(networkLayers);
            delta = Create2DLayers(networkLayers);
            signalError = Create1DLayers(networkLayers);
            output = CreateFull1DLayers(networkLayers);
            double dw0 = 0.20;
            for (int l = 1; l < numbersOfLayers; l++)
            {
                for (int n = 0; n < networkLayers[l]; n++)
                {
                    for (int w = 0; w < networkLayers[l-1]; w++)
                    {
                        weights[l][n, w] = 0.4 * (0.5 - rnd.NextDouble());
                        delta[l][n, w] = dw0;
                    }
                }
            }
        }
        private static double[][,] Create2DLayers(int[] networkLayers)
        {
            var result = new double[networkLayers.Length][,];
            result[(int)NetworkLayer.Input] = null;
            result[(int)NetworkLayer.Hidden] = new double[networkLayers[(int)NetworkLayer.Hidden], networkLayers[(int)NetworkLayer.Input] + 1];
            result[(int)NetworkLayer.Output] = new double[networkLayers[(int)NetworkLayer.Output], networkLayers[(int)NetworkLayer.Hidden] + 1];
            return result;
        }
        private static double[][] Create1DLayers(int[] networkLayers)
        {
            var result = new double[networkLayers.Length][];
            result[(int)NetworkLayer.Input] = null;
            result[(int)NetworkLayer.Hidden] = new double[networkLayers[(int)NetworkLayer.Hidden]];
            result[(int)NetworkLayer.Output] = new double[networkLayers[(int)NetworkLayer.Output]];
            return result;
        }
        private static double[][] CreateFull1DLayers(int[] networkLayers)
        {
            var result = new double[networkLayers.Length][];
            result[(int)NetworkLayer.Input] = new double[networkLayers[(int)NetworkLayer.Input]];
            result[(int)NetworkLayer.Hidden] = new double[networkLayers[(int)NetworkLayer.Hidden]];
            result[(int)NetworkLayer.Output] = new double[networkLayers[(int)NetworkLayer.Output]];
            return result;
        }
        public double Accuracy(int lok = 0)
        {
            double maxValue = -1;
            double error = 0.0;
            bool classification = false;
            if (baseData._trainingDataSet.GetLength(0)> networkLayers[0] + 1)
                classification = true;
            int numCorrect = 0;
            int maxIndex = -1;
            for (int v = 0; v < baseData._trainingDataSet.GetLength(0); v++)
            {
                Program.ForwardPass(this, v, lok);
                maxIndex = -1;
                maxValue = -1.1;
                for (int n = 0; n < networkLayers[numbersOfLayers - 1]; n++)
                {
                    if (classification)
                        error += GradientLearning.TransferFunction(this, output[numbersOfLayers - 1][n] - (2 * baseData._trainingDataSet[v,networkLayers[0] + n] - 1));
                    else
                        error += Math.Pow(output[numbersOfLayers - 1][n] - baseData._trainingDataSet[v,networkLayers[0] + n], 2);
                    if (output[numbersOfLayers - 1][n] > maxValue)
                    {
                        maxValue = output[numbersOfLayers - 1][n];
                        maxIndex = n;
                    }
                }
                int position = networkLayers[0] + maxIndex;
                if (baseData._trainingDataSet[v,position] == 1)
                    numCorrect++;
            }
            error /= baseData._trainingDataSet.GetLength(0);
            Console.WriteLine($"error {error}");
            return (double)numCorrect / baseData._trainingDataSet.GetLength(0);
        }
    }
}

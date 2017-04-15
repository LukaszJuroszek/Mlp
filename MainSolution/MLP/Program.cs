using MLPProgram.LearningAlgorithms;
using MLPProgram.Networks;
using System;
using System.Diagnostics;
using System.Linq;

namespace MLPProgram
{
    public static class Program
    {
        static void Main(string[] args)
        {
            string filePath = @"..\..\Datasets\ionosphere_std_sh.txt";
            var st = new Stopwatch();
            var testDataset = new FileParser(filePath, GradientLearning.SigmoidTransferFunction);
            var testDatasetNew = new FIleParserNew(filePath, GradientLearning.SigmoidTransferFunction);
            var data = new BaseDataHolder(testDataset);
            var dataNew = new DataHolder(testDatasetNew);
            var network = new MLP(data);
            var networkNew = new MLPNew(dataNew);
            var learningAlgorithm = new GradientLearning(network);
            var learningAlgorithmNew = new GradientLearningNew(networkNew);
            st.Start();
            for (var i = 0; i < 1; i++)
            {
                //to memory
                st.Reset();
                st.Start();
                learningAlgorithm.Train(numberOfEpochs: 50, batchSize: 30, learnRate: 0.05, momentum: 0.5);
                learningAlgorithmNew.Train(numberOfEpochs: 50, batchSize: 30, learnRate: 0.05, momentum: 0.5);
                double testAccuracy = network.Accuracy();
                Console.WriteLine(testAccuracy);
                double testAccuracyNew = networkNew.Accuracy();
                Console.WriteLine(testAccuracyNew);
                st.Stop();
                Console.WriteLine(st.Elapsed);
            }

        }
        public static void ForwardPass(MLP network, int indexOftrainingDataSet, int lok = -1)
        {
            network.output[0] = network.baseData._trainingDataSet[indexOftrainingDataSet].Take(network.output[0].Length).ToArray();
            for (int l = 1; l < network.output.Length; l++)
            {
                for (int n = 0; n < network.output[l].Length; n++)
                {
                    double sum = 0;
                    for (int w = 0; w < network.output[l - 1].Length; w++)
                    {
                        sum += network.output[l - 1][w] * network.weights[l][n][w];
                    }
                    sum += network.weights[l][n][network.output[l - 1].Length]; //bias
                    network.output[l][n] = (l == network.output.Length - 1 && !network.classification) ? sum : GradientLearning.TransferFunction(network, sum);
                }
            }
        }
        public static void ForwardPass(double[][] output, double[][] trainingDataSet, double[][][] weights, bool classification, bool isSigmoidFunction, int indexOftrainingDataSet, int lok = -1)
        {
            output[0] = trainingDataSet[indexOftrainingDataSet].Take(output[0].Length).ToArray();
            for (int l = 1; l < output.Length; l++)
            {
                for (int n = 0; n < output[l].Length; n++)
                {
                    double sum = 0;
                    for (int w = 0; w < output[l - 1].Length; w++)
                    {
                        sum += output[l - 1][w] * weights[l][n][w];
                    }
                    sum += weights[l][n][output[l - 1].Length]; //bias
                    output[l][n] = (l == output.Length - 1 && !classification) ? sum : GradientLearning.TransferFunction(isSigmoidFunction, sum);
                }
            }
        }
        public static void ForwardPass(MLPNew network, int indexOftrainingDataSet, int lok = -1)
        {
            for (int i = 0; i < network.networkLayers[0]; i++)
            {
                var p = network.baseData._trainingDataSet[indexOftrainingDataSet, i];
                network.output[(int)NetworkLayer.Input][i] = network.baseData._trainingDataSet[indexOftrainingDataSet, i];
            }
            for (int l = 1; l < network.numbersOfLayers; l++)
            {
                for (int n = 0; n < network.output[l].GetLength(0); n++)
                {
                    double sum = 0;
                    for (int w = 0; w < network.output[l - 1].GetLength(0); w++)
                    {
                        sum += network.output[l - 1][w] * network.weights[l][n, w];
                    }
                    sum += network.weights[l][n, network.output[l - 1].Length]; //bias
                    network.output[l][n] = (l == network.output[l].Length - 1 && !network.classification) ? sum : GradientLearning.TransferFunction(network, sum);

                }
            }
        }
    }

}

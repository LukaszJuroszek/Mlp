 using MLPProgram.LearningAlgorithms;
using MLPProgram.Networks;
using System;
using System.Collections.Generic;
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
            var mainNetwork = new MLPNew(dataNew,network.weights);
            var learningAlgorithm = new GradientLearning(network);
            st.Start();
            for (int i = 0; i < 1; i++)
            {
                //to memory
                st.Reset();
                st.Start();
                learningAlgorithm.Train(numberOfEpochs: 50, batchSize: 30, learnRate: 0.05, momentum: 0.5);
                double testAccuracy = network.Accuracy();
                Console.WriteLine(testAccuracy);
                st.Stop();
                Console.WriteLine(st.Elapsed);
            }
            int dataGroupCount = 351;
            var splitedDataGroups = DataHolder.GetTrainingDataAsChunks(mainNetwork.baseData._trainingDataSet, dataGroupCount);
            var trainingSystems = new TrainingSystem[splitedDataGroups.GetLength(0)];
            InitLearningSystemsBySplitedData(mainNetwork, splitedDataGroups, ref trainingSystems);
            for (int i = 0; i < trainingSystems.Length; i++)
            {
                st.Reset();
                st.Start();
                trainingSystems[i].TrainByInsideNetwork(numberOfEpochs: 50, batchSize: 30, learnRate: 0.05, momentum: 0.5);
                Console.WriteLine(trainingSystems[i]._network.CountAccuracyByTrainingData(mainNetwork.baseData._trainingDataSet));
            st.Stop();
            Console.WriteLine(st.Elapsed);
            }
        }
        private static void InitLearningSystemsBySplitedData(MLPNew mainNetwork, IEnumerable<double[]>[] splitedDataGroups, ref TrainingSystem[] learningSystems)
        {
            for (int i = 0; i < learningSystems.GetLength(0); i++)
            {
                var net = mainNetwork;
                net.baseData._trainingDataSet = splitedDataGroups[i].To2DArray();
                net.baseData._numberOfInputRow = net.baseData._trainingDataSet.GetLength(0)-1;
                learningSystems[i] = new TrainingSystem(net);
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
        public static void ForwardPass( double[][,]weights, int[] networkLayers,double[][] output,double[,] _trainingDataSet,int numbersOfLayers, bool classification, bool isSigmoidFunction, int indexOftrainingDataSet, int lok = -1)
        {
            for (int i = 0; i < networkLayers[0]; i++)
            {
               output[(int)NetworkLayer.Input][i] =_trainingDataSet[indexOftrainingDataSet, i];
            }
            for (int l = 1; l < numbersOfLayers; l++)
            {
                for (int n = 0; n < output[l].GetLength(0); n++)
                {
                    double sum = 0;
                    for (int w = 0; w < output[l - 1].GetLength(0); w++)
                    {
                        sum += output[l - 1][w] * weights[l][n, w];
                    }
                    sum += weights[l][n, output[l - 1].Length]; //bias
                    output[l][n] = (l == output[l].Length - 1 && !classification) ? sum : GradientLearning.TransferFunction(isSigmoidFunction, sum);
                }
            }
        }
        public static void ForwardPass(MLPNew network, int indexOftrainingDataSet, int lok = -1)
        {
            for (int i = 0; i < network.networkLayers[0]; i++)
            {
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
        public static T[][] ToJagged2DArray<T>(this T[,] source)
        {
            var reslut = new T[source.GetLength(0)][];

            for (int c = 0; c < source.GetLength(0); c++)
            {
                reslut[c] = new T[source.GetLength(1)];
                for (int r = 0; r < source.GetLength(1); r++)
                    reslut[c][r] = source[c, r];
            }
            return reslut;
        }
        public static T[,] To2DArray<T>(this IEnumerable<T[]> source)
        {
            var reslut = new T[source.Count(), source.First().Count()];
            for (int count = 0; count < source.Count(); count++)
                for (int r = 0; r < source.First().Count(); r++)
                    reslut[count, r] = source.ToArray()[count][r];
            return reslut;
        }
        public static IEnumerable<T[][]> SplitList<T>(T[][] array, int nSize = 30)
        {

            for (int i = 0; i < array.GetLength(0); i += nSize)
            {
                yield return array.Skip(i).Take(Math.Min(nSize, array.GetLength(0) - i)).ToArray();
            }
        }
    }
}

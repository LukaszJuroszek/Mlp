using Alea;
using MLPProgram.LearningAlgorithms;
using MLPProgram.Networks;
using System;
using System.Diagnostics;
using System.Linq;

namespace MLPProgram
{
    [GpuManaged]
    public static class Program
    {
        static void Main(string[] args)
        {
            var filePath = @"..\..\Datasets\ionosphere_std_sh.txt";
            var st = new Stopwatch();
            var totalMs = TimeSpan.FromMilliseconds(0);
            var testDataset = new FileParser(filePath, GradientLearning.SigmoidTransferFunction);
            var data = new BaseDataHolder(testDataset);
            var network = new MLP(data);
            var learningAlgorithm = new GradientLearning(network, data);
            st.Start();
            for (var i = 0; i < 1; i++)
            {
                //to memory
                st.Reset();
                st.Start();
                learningAlgorithm.Train(numberOfEpochs: 50, batchSize: 30, learnRate: 0.05, momentum: 0.5);
                //learningAlgorithm.TrainOld(numberOfEpochs: 50, batchSize: 30, learnRate: 0.05, momentum: 0.5);
                var testAccuracy = network.Accuracy(data,out double mseTrain);
                Console.WriteLine(testAccuracy);
                Console.WriteLine(mseTrain);
                st.Stop();
                Console.WriteLine(st.Elapsed);
            }
        }
        public static void ForwardPass(MLP network,BaseDataHolder baseData, int row, int lok = -1)
        {
            network.output[0] = baseData._trainingDataSet[row].Take(network.output[0].Length).ToArray();
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
                    network.output[l][n] = (l == network.output.Length - 1 && !network.classification) ? sum : GradientLearning.TransferFunction(baseData, sum);
                }
            }
        }
    }
}

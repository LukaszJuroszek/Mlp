using MLPProgram.LearningAlgorithms;
using MLPProgram.Networks;
using System;
using System.Diagnostics;
using System.Linq;

namespace MLPProgram
{
    class Program
    {
        static void Main(string[] args)
        {
            var filePath = @"..\..\Datasets\ImageSegmentation_std_sh.txt";
            var st = new Stopwatch();
            var totalMs = TimeSpan.FromMilliseconds(0);
            var testDataset = new FileParser(filePath, TransferFunctions.SigmoidTransferFunction);
            var data = new BaseDataHolder(testDataset);
            var network = new MLP(data);
            var learningAlgorithm = new GradientLearning(network);
            st.Start();
            for (var i = 0; i < 10; i++)
            {
                //to memory
                st.Reset();
                st.Start();
                learningAlgorithm.Train(numberOfEpochs: 50, batchSize: 30, learnRate: 0.05, momentum: 0.5);
                //Console.WriteLine(testAccuracy);
                //Console.WriteLine(mseTrain);
                var testAccuracy = network.Accuracy(out double mseTrain);
                st.Stop();
                Console.WriteLine(st.Elapsed);
            }

        }
        public static void ForwardPass(MLP network, int indexOftrainingDataSet, int lok = -1)
        {
            //coping trainingData to output[0]
            network.output[0] = network.baseData._trainingDataSet[indexOftrainingDataSet].Take(network.output[0].Length).ToArray();
            //calculate outputs by output[0]...==_trainingDataSet[indexOftrainingDataSet].
            for (var l = 1; l < network.output.Length; l++)
            {
                for (var n = 0; n < network.output[l].Length; n++)
                {
                    double sum = 0;
                    //l-1 means that we are taking prevouls op.layer...
                    for (var w = 0; w < network.output[l - 1].Length; w++)
                    {
                        sum += network.output[l - 1][w] * network.weights[l][n][w];
                    }
                    sum += network.weights[l][n][network.output[l - 1].Length]; //bias
                    network.output[l][n] = (l == network.output.Length - 1 && !network.classification) ? sum : network.baseData.TransferFunction(sum);
                }
            }
        }
    }
}

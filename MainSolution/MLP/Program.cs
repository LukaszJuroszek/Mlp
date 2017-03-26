using Alea;
using Alea.Parallel;
using MLPProgram.LearningAlgorithms;
using MLPProgram.Networks;
using System;
using System.Diagnostics;

namespace MLPProgram
{
    class Program
    {
        static void Main(string[] args)
        {
            var trainingFileName = @"..\..\Datasets\page-blocks_std_sh.txt";
            var testFile = @"..\..\Datasets\page-blocks_std_sh.txt";
            var st = new Stopwatch();
            st.Start();
            //to memory
            var trainingDataset = new DataFileHolder(trainingFileName, GradientLearning.SigmoidTransferFunction);
            st.Stop();
            Console.WriteLine(st.Elapsed);
            st.Reset();
            st.Start();
            var network = new MLP(trainingDataset.GetLayers(), trainingDataset.Classification, GradientLearning.SigmoidTransferFunction);
            var learningAlgorithm = new Rprop(network,trainingDataset);
            learningAlgorithm.Train(numEpochs: 50, batchSize: 30, learnRate: 0.05, momentum: 0.5);
            var testDataset = new DataFileHolder(testFile, GradientLearning.SigmoidTransferFunction);
            var testAccuracy = network.Accuracy(testDataset.Data, out double mseTrain, GradientLearning.SigmoidTransferFunction);
            st.Stop();
            Console.WriteLine(st.Elapsed);
            Console.WriteLine(testAccuracy);
            Console.WriteLine(mseTrain);
        }
        public static void ForwardPass(MLP _network,double[] vector, Func<double, double> transferFunction, int lok = -1)
        {
            {
                for (var i = 0; i < _network.layer[0]; i++)
                    _network.output[0][i] = vector[i];
            }
            for (var l = 1; l < _network.numLayers; l++)
            {
                for (var n = 0; n < _network.layer[l]; n++)
                {
                    double sum = 0;
                    for (var w = 0; w < _network.layer[l - 1]; w++)
                        sum += _network.output[l - 1][w] * _network.weights[l][n][w];
                    sum += _network.weights[l][n][_network.layer[l - 1]]; //bias
                    if (l == _network.numLayers - 1 && !_network.classification)
                        _network.output[l][n] = sum;
                    else
                        _network.output[l][n] = transferFunction(sum);
                }
            }
           
        }
    }
}

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
            var filePath = @"..\..\Datasets\mortgage_std_sh.txt";
            var st = new Stopwatch();
            var totalMs = TimeSpan.FromMilliseconds(0);
            var trainData = new FileParser(filePath, TransferFunctions.SigmoidTransferFunction);
            var testDataset = new FileParser(filePath, TransferFunctions.SigmoidTransferFunction);
            var data = new BaseDataHolder(trainData);
            var network = new MLP(data);
            var learningAlgorithm = new Rprop(network);
            st.Start();
            for (var i = 0; i < 3; i++)
            {
                //to memory
                //st.Stop();
                //Console.WriteLine(st.Elapsed);
                //st.Reset();
                //st.Start();
                learningAlgorithm.Train(numEpochs: 50, batchSize: 30, learnRate: 0.05, momentum: 0.5);
                var testAccuracy = network.Accuracy(out double mseTrain);
            }
            //Console.WriteLine(testAccuracy);
            //Console.WriteLine(mseTrain);
            st.Stop();
            Console.WriteLine(st.Elapsed);

        }
        public static void ForwardPass(MLP _network, double[] vector, BaseDataHolder transferFuncFromBaseData, int lok = -1)
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
                        _network.output[l][n] = transferFuncFromBaseData.TransferFunction(sum);
                }
            }

        }
    }
}

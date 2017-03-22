using MLPProgram.LearningAlgorithms;
using MLPProgram.Networks;
using MLPProgram.TransferFunctions;
using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace MLPProgram
{
    class Program
    {
        static void Main(string[] args)
        {
            var trainingFileName = @"..\..\Datasets\iris_std_sh.txt";
            var testFile = @"..\..\Datasets\iris_std_sh.txt";
            var st = new Stopwatch();
            st.Start();
            var transferFunction = new HyperbolicTangent();
            //to memory
            var trainingDataset = new DataFileHolder(trainingFileName, transferFunction);
            var network = new MLP(trainingDataset.GetLayers(), trainingDataset.Classification, transferFunction);
            var learningAlgorithm = new Rprop(network);
            learningAlgorithm.Train(trainingDataset.Data, trainingDataset.Classification, numEpochs: 50, batchSize: 30, learnRate: 0.05, momentum: 0.5);
            var testDataset = new DataFileHolder(testFile, transferFunction);
            var testAccuracy = network.Accuracy(testDataset.Data, out double mseTrain, transferFunction);
            st.Stop();
            Console.WriteLine(st.Elapsed);
            Console.WriteLine(testAccuracy);
            Console.WriteLine(mseTrain);
        }
    }
}

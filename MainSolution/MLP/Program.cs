using MLPProgram.LearningAlgorithms;
using MLPProgram.Networks;
using MLPProgram.TransferFunctions;
using System;
using System.Collections.Generic;
namespace MLPProgram
{
    class Program
    {
        static void Main(string[] args)
        {
            string trainingFile = @"..\..\Datasets\iris_std_sh.txt";
            string testFile = @"..\..\Datasets\iris_std_sh.txt";
            ITransferFunction transferFunction = new HyperbolicTangent();
           // TransferFunctions.ITransferFunction transferFunction = new TransferFunctions.Sigmoid();
            double[][] trainingDataset = Utils.LoadFile(trainingFile, out string headerLine, out int numInput, out int numOutput,out bool classification, transferFunction);
            int[] numHidden = new int[] { (int)Math.Sqrt(numInput * numOutput) };
            List<int> LL = new List<int>();
            LL.Add(numInput);
            LL.AddRange(numHidden);
            LL.Add(numOutput);
            int[] Layers = LL.ToArray();
            INetwork network = new MLP(Layers, classification, transferFunction);
           // LearningAlgorithms.ILearningAlgorithm learningAlgorithm = new LearningAlgorithms.BP();
            ILearningAlgorithm learningAlgorithm = new Rprop();
            learningAlgorithm.Train(network, trainingDataset, classification, numEpochs:50, batchSize:30, learnRate:0.05, momentum:0.5);
            double[][] testDataset = Utils.LoadFile(testFile, out headerLine, out numInput, out numOutput,out classification, transferFunction);
            double testAccuracy = network.Accuracy(testDataset, out double mseTrain, transferFunction);
            Console.WriteLine(testAccuracy);
            Console.ReadLine();
        }
    }
}

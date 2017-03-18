using System;
namespace DL.LearningAlgorithms
{
    abstract class GradientLearning : ILearningAlgorithm
    {
        public double etaPlus = 1.2, etaMinus = 0.5, minDelta = 0.00001, maxDelta = 10, errorExponent = 1.0;
        public bool cv { get; set; }
        public int numSelectedVectors { get; set; }
        Networks.MLP network;
        public double Test(double[][] TrainingDataSet,double[][] TestDataSet)
        {
            double ErrorsRMSE;
            return network.Accuracy(TestDataSet,out ErrorsRMSE,network.transferFunction,0);
        }
        public void Train(Networks.INetwork network1,double[][] TrainingDataSet,ErrorFunctions.IErrorFunction errorFunction,DataSelection.IDataSelection dataSelection,
            bool classification,int numEpochs = 1000,int batchSize = 15,double learnRate = 0.05,double momentum = 0.2,int numRemovedFeatures = 0,double minError = -1,double maxError = 1000)
        {
            network = (Networks.MLP)network1;
            int numInputs = network.Layer[0];
            int numOutputs = network.Layer[network.numLayers - 1];
            int numVectors = TrainingDataSet.Length;
            if (batchSize > numVectors)
                batchSize = numVectors;
            if (this is Rprop)
                batchSize = numVectors;
            // int maxDegreeOfParallelism = Math.Max(1,(batchSize * network.numWeights) / 250);
            bool elim = false;
            int epoch = 0;
            double derivative = 0.0;
            for (int L = 1;L < network.numLayers;L++)
                for (int n = 0;n < network.Layer[L];n++)
                    for (int w = 0;w <= network.Layer[L - 1];w++)
                    {
                        network.weightDiff[L][n][w] = 0;
                        network.Delta[L][n][w] = 0.1;
                    }
            double numEpochs2 = 0.09 * numEpochs * numEpochs;
            double nep = 0;
            double minError0 = -1;
            double maxError0 = 10000;
            double epx = 0;
            while (epoch < numEpochs) // main training loop
            {
                epoch++;
                for (int L = 1;L < network.numLayers;L++)
                    for (int n = 0;n < network.Layer[L];n++)
                        for (int w = 0;w <= network.Layer[L - 1];w++)
                            network.weightDiff[L][n][w] = 0;
                if (epoch > 0.7 * numEpochs)
                {
                    epx++;
                    nep = ( epx * epx ) / numEpochs2;
                    minError0 = minError * nep;
                    maxError0 = maxError / nep;
                }
                double sum;
                int v = 0;
                int nsa = TrainingDataSet[0].Length - 2;
                numSelectedVectors = numVectors;
                while (v < numVectors)
                {
                    for (int b = 0;b < batchSize;b++)
                    {
                        network.ForwardPass(TrainingDataSet[v],network.transferFunction);
                        // find SignalErrors for the output layer
                        if (TrainingDataSet[v][nsa] != 0)
                        {
                            double sumError = 0;
                            for (int n = 0;n < numOutputs;n++)
                            {
                                double error = ( TrainingDataSet[v][numInputs + n] - network.output[network.numLayers - 1][n] );
                                error = Math.Sign(error) * Math.Pow(Math.Abs(error),errorExponent);

                                // if (dataSelection != null)
                                error *= TrainingDataSet[v][nsa];
                                sumError += Math.Abs(error);

                                if (classification)
                                {
                                    derivative = network.transferFunction.Derivative(network.output[network.numLayers - 1][n]);
                                } else
                                    derivative = 1.0;

                                network.SignalError[network.numLayers - 1][n] = error * derivative;
                            }

                            if (sumError <= minError0 || sumError >= maxError0)
                            {
                                TrainingDataSet[v][nsa] = 0;
                                numSelectedVectors--;
                            }
                        } else
                        {
                            numSelectedVectors--;
                        }
                        // find SignalErrors for all hidden layers
                        for (int L = network.numLayers - 2;L > 0;L--)
                        {
                            for (int n = 0;n < network.Layer[L];n++)
                            {
                                sum = 0.0;
                                for (int w = 0;w < network.Layer[L + 1];w++)
                                    sum += network.SignalError[L + 1][w] * network.Weights[L + 1][w][n];

                                derivative = network.transferFunction.Derivative(network.output[L][n]);
                                network.SignalError[L][n] = derivative * sum;
                            }
                        }
                        for (int L = network.numLayers - 1;L > 0;L--)
                        {
                            for (int n = 0;n < network.Layer[L];n++)
                            {
                                //bias
                                network.weightDiff[L][n][network.Layer[L - 1]] += learnRate * network.SignalError[L][n];

                                for (int w = 0;w < network.Layer[L - 1];w++)
                                    network.weightDiff[L][n][w] += learnRate * network.SignalError[L][n] * network.output[L - 1][w];
                            }
                        }
                        v++;
                        if (v == numVectors)
                            break;
                    }
                    updateWeights(network,learnRate,momentum,etaPlus,etaMinus,minDelta,maxDelta,numRemovedFeatures);
                    // zero-out gradients
                    for (int L = 1;L < network.numLayers;L++)
                        for (int n = 0;n < network.Layer[L];n++)
                            for (int w = 0;w <= network.Layer[L - 1];w++)
                                network.weightDiff[L][n][w] = 0;
                }
            }
        }
        abstract protected void updateWeights(Networks.MLP network,double learnRate,double momentum,double etaPlus,double etaMinus,double minDelta,double maxDelta,double inputWeightRegularizationCoef = -1);
    }
}

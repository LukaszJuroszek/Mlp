using MLPProgram.LearningAlgorithms;
using System;
using System.Text;

namespace MLPProgram.Networks
{
    public struct MLP
    {
        public double[][][] weightDiff, prevWeightDiff, delta, weights;
        public double[][] signalError, output;
        public int[] layer;
        public int numbersOfLayers;
        public bool classification;
        public int numWeights;
        public BaseDataHolder baseData;
        public override string ToString()
        {
            var st = new StringBuilder();
            st.Append($"Weights {numWeights} ");
            st.Append($"classification {classification} ");
            st.Append($"LayersCount {numbersOfLayers} ");
            for (int i = 0; i < layer.Length; i++)
            {
                st.Append($"l[{i}]={layer[i]} ");
            }
            return st.ToString();
        }
        public MLP(BaseDataHolder data, string weightFile = "")
        {
            baseData = data;
            classification = data._classification;
            layer = data._layer;
            numWeights = 0;
            numbersOfLayers = layer.Length;
            weights = new double[numbersOfLayers][][];
            weightDiff = new double[numbersOfLayers][][];
            delta = new double[numbersOfLayers][][];
            signalError = new double[numbersOfLayers][];
            output = new double[numbersOfLayers][];
            output[0] = new double[layer[0]];
            prevWeightDiff = new double[numbersOfLayers][][];
            var rnd = new Random();
            for (int l = 1; l < numbersOfLayers; l++)
            {
                weights[l] = new double[layer[l]][];
                weightDiff[l] = new double[layer[l]][];
                prevWeightDiff[l] = new double[layer[l]][];
                delta[l] = new double[layer[l]][];
                signalError[l] = new double[layer[l]];
                output[l] = new double[layer[l]];
                for (int n = 0; n < layer[l]; n++)
                {
                    weights[l][n] = new double[layer[l - 1] + 1];
                    weightDiff[l][n] = new double[layer[l - 1] + 1];
                    prevWeightDiff[l][n] = new double[layer[l - 1] + 1];
                    delta[l][n] = new double[layer[l - 1] + 1];
                    numWeights++;
                }
            }
            double dw0 = 0.20;
            for (int l = 1; l < numbersOfLayers; l++)
                for (int n = 0; n < layer[l]; n++)
                    for (int w = 0; w < layer[l - 1] + 1; w++)
                    {
                        weights[l][n][w] = 0.4 * (0.5 - rnd.NextDouble());//create random weigths 
                        delta[l][n][w] = dw0; //for Rprop
                    }
        }
        public static double CountAccuracy(MLP network, int lok = 0)
        {
            double maxValue = -1;
            double error = 0.0;
            bool classification = false;
            if (network.baseData._trainingDataSet[0].Length > network.layer[0] + 1)
                classification = true;
            int numCorrect = 0;
            int maxIndex = -1;
            for (int v = 0; v < network.baseData._trainingDataSet.Length; v++)
            {
                Program.ForwardPass(network, v, lok);
                maxIndex = -1;
                maxValue = -1.1;
                for (int n = 0; n < network.layer[network.numbersOfLayers - 1]; n++)
                {
                    if (classification)
                        error += GradientLearning.TransferFunction(network, network.output[network.numbersOfLayers - 1][n] - (2 * network.baseData._trainingDataSet[v][network.layer[0] + n] - 1));
                    else
                        error += Math.Pow(network.output[network.numbersOfLayers - 1][n] - network.baseData._trainingDataSet[v][network.layer[0] + n], 2);
                    if (network.output[network.numbersOfLayers - 1][n] > maxValue)
                    {
                        maxValue = network.output[network.numbersOfLayers - 1][n];
                        maxIndex = n;
                    }
                }
                int position = network.layer[0] + maxIndex;
                if (network.baseData._trainingDataSet[v][position] == 1)
                    numCorrect++;
            }
            error /= network.baseData._trainingDataSet.GetLength(0);
            Console.WriteLine($"error {error}");
            return (double)numCorrect / network.baseData._trainingDataSet.GetLength(0);
        }
        public static double Accuracy(MLP mlp, out double error, int lok = 0)
        {
            double maxValue = -1;
            error = 0.0;
            bool classification = false;
            if (mlp.baseData._trainingDataSet[0].Length > mlp.layer[0] + 1)
                classification = true;
            int numCorrect = 0;
            int maxIndex = -1;
            for (int v = 0; v < mlp.baseData._trainingDataSet.Length; v++)
            {
                Program.ForwardPass(mlp, v, lok);
                maxIndex = -1;
                maxValue = -1.1;
                for (int n = 0; n < mlp.layer[mlp.numbersOfLayers - 1]; n++)
                {
                    if (classification)
                        error += GradientLearning.TransferFunction(mlp, mlp.output[mlp.numbersOfLayers - 1][n] - (2 * mlp.baseData._trainingDataSet[v][mlp.layer[0] + n] - 1));
                    else
                        error += Math.Pow(mlp.output[mlp.numbersOfLayers - 1][n] - mlp.baseData._trainingDataSet[v][mlp.layer[0] + n], 2);
                    if (mlp.output[mlp.numbersOfLayers - 1][n] > maxValue)
                    {
                        maxValue = mlp.output[mlp.numbersOfLayers - 1][n];
                        maxIndex = n;
                    }
                }
                int position = mlp.layer[0] + maxIndex;
                if (mlp.baseData._trainingDataSet[v][position] == 1)
                    numCorrect++;
            }
            error /= mlp.baseData._trainingDataSet.Length;
            return (double)numCorrect / mlp.baseData._trainingDataSet.Length;
        }
        public double[] GetNonSignalErrorTable(double[][] DataSet, ref double accuracy, double errorExponent = 2.0)
        {
            int numVect = DataSet.Length;
            var errorTable = new double[numVect];
            double error = 0;
            for (int v = 0; v < numVect; v++)
            {
                error = 0;
                for (int n = 0; n < layer[0]; n++)
                    output[0][n] = DataSet[v][n];
                for (int l = 1; l < numbersOfLayers; l++)
                {
                    for (int n = 0; n < layer[l]; n++)
                    {
                        double sum = 0;
                        for (int w = 0; w < layer[l - 1]; w++)
                            sum += output[l - 1][w] * weights[l][n][w];
                        sum += weights[l][n][layer[l - 1]];
                        if (l == numbersOfLayers - 1 && !classification)
                            output[l][n] = sum;
                        else
                            output[l][n] = GradientLearning.TransferFunction(this, sum);
                        if (l == numbersOfLayers - 1)
                            error += Math.Pow(Math.Abs(output[l][n] - DataSet[v][layer[0] + n]), errorExponent);
                    }
                }
                errorTable[v] = error;
            }
            return errorTable;
        }
    }
}

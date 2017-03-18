using System;

namespace NeutralNetworks.Networks
{
    class MLP : INetwork
    {
        public double[][][] Weights, weightDiff, prevWeightDiff, Delta;
        public double[][] output, SignalError;
        public int[] Layer;
        public int numLayers, numWeights;
        public bool classification;
        public TransferFunctions.ITransferFunction transferFunction;
        public double[] featureImportance;
        public int[] featureNumber;
        public MLP(int[] Layer,bool classification,TransferFunctions.ITransferFunction transferFunction,string weightFile = "")
        {
            this.classification = classification;
            this.Layer = Layer;
            this.transferFunction = transferFunction;
            numLayers = Layer.Length;
            Weights = new double[numLayers][][];
            weightDiff = new double[numLayers][][];
            prevWeightDiff = new double[numLayers][][];
            Delta = new double[numLayers][][];
            SignalError = new double[numLayers][];
            output = new double[numLayers][];
            numWeights = 0;
            for (int L = 1;L < numLayers;L++)
            {
                Weights[L] = new double[Layer[L]][];
                weightDiff[L] = new double[Layer[L]][];
                prevWeightDiff[L] = new double[Layer[L]][];
                Delta[L] = new double[Layer[L]][];
                SignalError[L] = new double[Layer[L]];
                output[L] = new double[Layer[L]];
                for (int n = 0;n < Layer[L];n++)
                {
                    Weights[L][n] = new double[Layer[L - 1] + 1];
                    weightDiff[L][n] = new double[Layer[L - 1] + 1];
                    prevWeightDiff[L][n] = new double[Layer[L - 1] + 1];
                    Delta[L][n] = new double[Layer[L - 1] + 1];
                    numWeights++;
                }
            }
            output[0] = new double[Layer[0]];
            var dw0 = 0.20;
            var rnd = new Random();
            for (int L = 1;L < numLayers;L++)
                for (int n = 0;n < Layer[L];n++)
                    for (int w = 0;w < Layer[L - 1] + 1;w++)
                    {
                        Weights[L][n][w] = 0.4 * ( 0.5 - rnd.NextDouble() );
                        Delta[L][n][w] = dw0; //for VSS and Rprop
                    }
        }
        public double Accuracy(double[][] DataSet,out double error,TransferFunctions.ITransferFunction transferFunction,int lok = 0)
        {
            double maxValue = -1;
            error = 0.0;
            var classification = false;
            if (DataSet[0].Length > Layer[0] + 1)
                classification = true;
            int numCorrect = 0, maxIndex = -1;
            for (int v = 0;v < DataSet.Length;v++)
            {
                ForwardPass(DataSet[v],transferFunction,lok);
                maxIndex = -1;
                maxValue = -1.1;
                for (int n = 0;n < Layer[numLayers - 1];n++)
                {
                    if (classification)
                        error += transferFunction.TransferFunction(output[numLayers - 1][n] - ( 2 * DataSet[v][Layer[0] + n] - 1 ));
                    else
                        error += Math.Pow(output[numLayers - 1][n] - DataSet[v][Layer[0] + n],2);
                    if (output[numLayers - 1][n] > maxValue)
                    {
                        maxValue = output[numLayers - 1][n];
                        maxIndex = n;
                    }
                }
                int position = Layer[0] + maxIndex;
                if (DataSet[v][position] == 1)
                    numCorrect++;
            }
            error /= DataSet.Length;
            return (double)numCorrect / DataSet.Length;
        }
        public void ForwardPass(double[] vector,TransferFunctions.ITransferFunction transferFunction,int lok = -1)
        {
            for (int i = 0;i < Layer[0];i++)
                output[0][i] = vector[i];
            for (int L = 1;L < numLayers;L++)
            {
                for (int n = 0;n < Layer[L];n++)
                {
                    double sum = 0;
                    for (int w = 0;w < Layer[L - 1];w++)
                    {
                        sum += output[L - 1][w] * Weights[L][n][w];
                    }
                    sum += Weights[L][n][Layer[L - 1]]; //bias

                    if (L == numLayers - 1 && !classification)
                        output[L][n] = sum;
                    else
                        output[L][n] = transferFunction.TransferFunction(sum);
                }
            }
        }
        public double[] GetNonSignalErrorTable(double[][] DataSet,ref double accuracy,DataSelection.IDataSelection dataSelection,double errorExponent = 2.0)
        {
            int numVect = DataSet.Length;
            double[] ErrorTable = new double[numVect];
            double error = 0;

            for (int v = 0;v < numVect;v++)
            {
                error = 0;
                for (int n = 0;n < Layer[0];n++)
                    output[0][n] = DataSet[v][n];

                for (int L = 1;L < numLayers;L++)
                {
                    for (int n = 0;n < Layer[L];n++)
                    {
                        double sum = 0;
                        for (int w = 0;w < Layer[L - 1];w++)
                            sum += output[L - 1][w] * Weights[L][n][w];

                        sum += Weights[L][n][Layer[L - 1]];

                        if (L == numLayers - 1 && !classification)
                            output[L][n] = sum;
                        else
                            output[L][n] = transferFunction.TransferFunction(sum);

                        if (L == numLayers - 1)
                            error += Math.Pow(Math.Abs(output[L][n] - DataSet[v][Layer[0] + n]),errorExponent);
                    }
                }
                ErrorTable[v] = error;
            }
            if (dataSelection != null)
            {
                int outlierColumn = DataSet[0].Length - 2;
                for (int v = 0;v < numVect;v++)
                    ErrorTable[v] *= DataSet[v][outlierColumn];
            }
            return ErrorTable;
        }
    }
}

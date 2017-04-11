using Alea;
using MLPProgram.LearningAlgorithms;
using System;
using System.Text;

namespace MLPProgram.Networks
{
    public struct MLP
    {
        //public double[] featureImportance;
        //public int[] featureNumber;
        //[GpuParam]
        //public Random rnd;
        [GpuParam]
        public double[][][] weightDiff;
        [GpuParam]
        public double[][][] prevWeightDiff;
        [GpuParam]
        public double[][][] delta;
        [GpuParam]
        public double[][][] weights;
        [GpuParam]
        public double[][] signalError;
        [GpuParam]
        public double[][] output;
        [GpuParam]
        public int[] layer;
        [GpuParam]
        public int numberOfLayer;
        [GpuParam]
        public bool classification;
        [GpuParam]
        public int numberOfWeigth;
        public override string ToString()
        {
            var st = new StringBuilder();
            //st.Append($"Weights {numberOfWeigth} ");
            //st.Append($"classification {classification} ");
            //st.Append($"LayersCount {numberOfLayer} ");
                st.AppendLine($"l[{1}]={weights[1][0][1]} ");
                st.AppendLine($"l[{1}]={weights[1][0][2]} ");
                st.AppendLine($"l[{1}]={weights[1][1][1]} ");
                st.AppendLine($"l[{1}]={weights[1][1][2]} ");
            st.AppendLine($"l[{1}]={output[1][0]} ");
            st.AppendLine($"l[{1}]={output[1][0]} ");
            st.AppendLine($"l[{1}]={output[1][1]} ");
            st.AppendLine($"l[{1}]={output[1][1]} ");
            return st.ToString();
        }
        public MLP(BaseDataHolder data, string weightFile = "")
        {
            classification = data._classification;
            layer = data._layer;
            numberOfWeigth = 0;
            numberOfLayer = layer.Length;
            weights = new double[numberOfLayer][][];
            weightDiff = new double[numberOfLayer][][];
            delta = new double[numberOfLayer][][];
            signalError = new double[numberOfLayer][];
            output = new double[numberOfLayer][];
            output[0] = new double[layer[0]];
            prevWeightDiff = new double[numberOfLayer][][];
            var rnd = new Random();
            InitMultiDimArray(layer);
            var dw0 = 0.20;
            for (var l = 1; l < numberOfLayer; l++)
                for (var n = 0; n < layer[l]; n++)
                    for (var w = 0; w < layer[l - 1] + 1; w++)
                    {
                        weights[l][n][w] = 0.4 * (0.5 - rnd.NextDouble());//create random weigths 
                        delta[l][n][w] = dw0; //for Rprop
                    }
        }
        private void InitMultiDimArray(int[] layer)
        {
            for (var l = 1; l < numberOfLayer; l++)
            {
                weights[l] = new double[layer[l]][];
                weightDiff[l] = new double[layer[l]][];
                prevWeightDiff[l] = new double[layer[l]][];
                delta[l] = new double[layer[l]][];
                signalError[l] = new double[layer[l]];
                output[l] = new double[layer[l]];
                for (var n = 0; n < layer[l]; n++)
                {
                    weights[l][n] = new double[layer[l - 1] + 1];
                    weightDiff[l][n] = new double[layer[l - 1] + 1];
                    prevWeightDiff[l][n] = new double[layer[l - 1] + 1];
                    delta[l][n] = new double[layer[l - 1] + 1];
                    numberOfWeigth++;
                }
            }
        }
        public double Accuracy(BaseDataHolder baseData, out double error, int lok = 0)
        {
            double maxValue = -1;
            error = 0.0;
            var classification = false;
            if (baseData._trainingDataSet[0].Length > layer[0] + 1)
                classification = true;
            var numCorrect = 0;
            var maxIndex = -1;
            for (var v = 0; v < baseData._trainingDataSet.Length; v++)
            {
                Program.ForwardPass(this, baseData, v, lok);
                maxIndex = -1;
                maxValue = -1.1;
                for (var n = 0; n < layer[numberOfLayer - 1]; n++)
                {
                    if (classification)
                    {
                        var value = output[numberOfLayer - 1][n] - (2 * baseData._trainingDataSet[v][layer[0] + n] - 1);
                        error += GradientLearning.TransferFunction(baseData, value);
                    }
                    else
                        error += Math.Pow(output[numberOfLayer - 1][n] - baseData._trainingDataSet[v][layer[0] + n], 2);
                    if (output[numberOfLayer - 1][n] > maxValue)
                    {
                        maxValue = output[numberOfLayer - 1][n];
                        maxIndex = n;
                    }
                }
                var position = layer[0] + maxIndex;
                if (baseData._trainingDataSet[v][position] == 1)
                    numCorrect++;
            }
            error /= baseData._trainingDataSet.Length;
            return (double)numCorrect / baseData._trainingDataSet.Length;
        }

        public double[] GetNonSignalErrorTable(double[][] DataSet, ref double accuracy, double errorExponent = 2.0)
        {
            var numVect = DataSet.Length;
            double[] errorTable = new double[numVect];
            double error = 0;
            for (var v = 0; v < numVect; v++)
            {
                error = 0;
                for (var n = 0; n < layer[0]; n++)
                    output[0][n] = DataSet[v][n];
                for (var l = 1; l < numberOfLayer; l++)
                {
                    for (var n = 0; n < layer[l]; n++)
                    {
                        double sum = 0;
                        for (var w = 0; w < layer[l - 1]; w++)
                            sum += output[l - 1][w] * weights[l][n][w];
                        sum += weights[l][n][layer[l - 1]];
                        if (l == numberOfLayer - 1 && !classification)
                            output[l][n] = sum;
                        else
                        //output[l][n] = GradientLearning.TransferFunction(, sum); //todo change transfer function
                        if (l == numberOfLayer - 1)
                            error += Math.Pow(Math.Abs(output[l][n] - DataSet[v][layer[0] + n]), errorExponent);
                    }
                }
                errorTable[v] = error;
            }
            return errorTable;
        }
    }
}

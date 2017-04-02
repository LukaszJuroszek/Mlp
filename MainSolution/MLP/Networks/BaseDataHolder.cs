using Alea;
using MLPProgram.LearningAlgorithms;
using System;
using System.Text;

namespace MLPProgram.Networks
{
    public struct BaseDataHolder
    {
        public double[][] _trainingDataSet;
        public int _numberOfInput;
        public int _numberOfOutput;
        public int _numberOFVectors;
        public bool _classification;
        public bool _isSigmoidFunction;
        public int[] _layer;
        public override string ToString()
        {
            var st = new StringBuilder();
            st.Append($"numberOfInput {_numberOfInput} ");
            st.Append($"numberOfOutput {_numberOfOutput} ");
            st.Append($"numberOFVector {_numberOFVectors} ");
            st.AppendLine($"classification {_classification} ");
            for (var i = 0; i < _trainingDataSet.Length; i++)
            {
                for (var pi = 0; pi < _trainingDataSet[i].Length; pi++)
                {
                    st.Append($"d[{i},{pi}] {_trainingDataSet[i][pi]:n4}, ");
                }
            }
            return st.ToString();
        }
        public BaseDataHolder(double[][] data, int numberOfInput, int numberOfOutput, int numberOFVectors, Func<double, double> transferFunction, bool classification, int[] layer)
        {
            _trainingDataSet = data;
            _layer = layer;
            _numberOfInput = numberOfInput;
            _numberOfOutput = numberOfOutput;
            _numberOFVectors = numberOFVectors;
            _classification = classification;
            _isSigmoidFunction = TransferFunctions.IsSigmoidTransferFunction(transferFunction);
        }
        public BaseDataHolder(FileParser file)
        {
            _trainingDataSet = file.Data;
            _layer = file.GetLayers();
            _numberOfInput = file.NumberOfInput;
            _numberOfOutput = file.NumberOfOutput;
            _numberOFVectors = file.NumberOFVectors;
            _classification = file.Classification;
            _isSigmoidFunction = TransferFunctions.IsSigmoidTransferFunction(file.TransferFunction);
        }
        public double TransferFunction(double x)
        {
            double result = 0;
            if (_isSigmoidFunction)
                result = TransferFunctions.SigmoidTransferFunction(x);
            else
                result = TransferFunctions.HyperbolicTransferFunction(x);
            return result;
        }
        public double DerivativeFunction(double x)
        {
            double result = 0;
            if (_isSigmoidFunction)
                result = TransferFunctions.SigmoidDerivative(x);
            else
                result = TransferFunctions.HyperbolicDerivative(x);
            return result;
        }
    }
}

using Alea;
using MLPProgram.LearningAlgorithms;
using System;
using System.Collections.Generic;
using System.Linq;
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
            _isSigmoidFunction = GradientLearning.IsSigmoidTransferFunction(transferFunction);
        }
        public BaseDataHolder(FileParser file)
        {
            _trainingDataSet = file.Data;
            _layer = file.GetLayers();
            _numberOfInput = file.NumberOfInput;
            _numberOfOutput = file.NumberOfOutput;
            _numberOFVectors = file.NumberOfInputRow;
            _classification = file.Classification;
            _isSigmoidFunction = GradientLearning.IsSigmoidTransferFunction(file.TransferFunction);
        }
    }
    public struct DataHolder
    {
        public double[,] _trainingDataSet;
        public int _numberOfInput;
        public int _numberOfOutput;
        public int _numberOfInputRow;
        public byte _classification;
        public byte _isSigmoidFunction;
        public int[] _layer;
        public DataHolder(double[,] data, int numberOfInput, int numberOfOutput, int numberOFVectors, Func<double, double> transferFunction, byte classification, int[] layer)
        {
            _trainingDataSet = data;
            _layer = layer;
            _numberOfInput = numberOfInput;
            _numberOfOutput = numberOfOutput;
            _numberOfInputRow = numberOFVectors;
            _classification = classification;
            _isSigmoidFunction = GradientLearning.IsSigmoidTransferFunctionByte(transferFunction);
        }
        public DataHolder(FIleParserNew file)
        {
            _trainingDataSet = file.Data;
            _layer = file.GetLayers();
            _numberOfInput = file.NumberOfInput;
            _numberOfOutput = file.NumberOfOutput;
            _numberOfInputRow = file.NumberOfInputRow;
            _classification = file.Classification;
            _isSigmoidFunction = GradientLearning.IsSigmoidTransferFunctionByte(file.TransferFunction);
        }

        public static IEnumerable<T[]>[] GetTrainingDataAsChunks<T>(T[,] array, int howManyGroup)
        {
            var splitedTraingData = Program.SplitList(array.ToJagged2DArray(), howManyGroup);
            var result = splitedTraingData.Select(x => x.Select(p => p).Where(p => p != null));
            return result.ToArray();
        }
    }
}

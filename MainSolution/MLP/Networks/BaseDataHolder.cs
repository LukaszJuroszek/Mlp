using Alea;
using MLPProgram.LearningAlgorithms;
using System;

namespace MLPProgram.Networks
{
    public struct BaseDataHolder
    {
        [GpuParam]
        public double[][] _data;
        [GpuParam]
        public int _numberOfInput;
        [GpuParam]
        public int _numberOfOutput;
        [GpuParam]
        public int _numberOFVectors;
        [GpuParam]
        public bool _classification;
        [GpuParam]
        public int[] _layer;
        [GpuParam]
        public bool _isSigmoidFunction;
        public BaseDataHolder(double[][] data, int numberOfInput, int numberOfOutput, int numberOFVectors, Func<double, double> transferFunction, bool classification, int[] layer)
        {
            _data = data;
            _layer = layer;
            _numberOfInput = numberOfInput;
            _numberOfOutput = numberOfOutput;
            _numberOFVectors = numberOFVectors;
            _classification = classification;
            _isSigmoidFunction = TransferFunctions.IsSigmoidTransferFunction(transferFunction);
        }
        public BaseDataHolder(FileParser file)
        {
            _data = file.Data;
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
            {
                result = TransferFunctions.SigmoidTransferFunction(x);
            }
            else
            {
                result = TransferFunctions.HyperbolicTransferFunction(x);
            }
            return result;
        }
        public double DerivativeFunction(double x)
        {
            double result = 0;
            if (_isSigmoidFunction)
            {
                result = TransferFunctions.SigmoidDerivative(x);
            }
            else
            {
                result = TransferFunctions.HyperbolicDerivative(x);
            }
            return result;
        }
    }
}

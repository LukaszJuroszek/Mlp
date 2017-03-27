using System;

namespace MLPProgram.Networks
{
    public struct BaseDataHolder
    {
        public double[][] _data;
        public int _numberOfInput;
        public int _numberOfOutput;
        public int _numberOFVectors;
        public bool _classification;
        public int[] _layer;
        public Func<double, double> _transferFunction;
        public BaseDataHolder(double[][] data, int numberOfInput, int numberOfOutput, int numberOFVectors, Func<double, double> transferFunction, bool classification, int[] layer)
        {
            _data = data;
            _layer = layer;
            _numberOfInput = numberOfInput;
            _numberOfOutput = numberOfOutput;
            _numberOFVectors = numberOFVectors;
            _transferFunction = transferFunction;
            _classification = classification;
        }
        public BaseDataHolder(FileParser file)
        {
            _data = file.Data;
            _layer = file.GetLayers();
            _numberOfInput = file.NumberOfInput;
            _numberOfOutput = file.NumberOfOutput;
            _numberOFVectors = file.NumberOFVectors;
            _transferFunction = file.TransferFunction;
            _classification = file.Classification;
        }
    }
}

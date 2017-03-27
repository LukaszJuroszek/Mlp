using Alea;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
namespace MLPProgram
{
    public struct DataFileHolder
    {
        [GpuParam]
        public string _headerLine ;
        [GpuParam]
        public int _numberOfInput ;
        [GpuParam]
        public int _numberOfOutput ;
        [GpuParam]
        public int _numberOFVectors ;
        [GpuParam]
        public int _numberOfAttributes ;
        [GpuParam]
        public string[] _headers ;
        [GpuParam]
        public bool _classification ;
        [GpuParam]
        public double[][] _data ;
        [GpuParam]
        public Func<double,double> _transferFunction;
        public DataFileHolder(string fileName, Func<double, double> transferFunction, bool multipleClassColumns = true, bool firstStandardizeRun = false)
        {
            _transferFunction = transferFunction;
            using (var sr = new StreamReader(fileName))
            {
                _numberOFVectors=0;
                _headerLine = sr.ReadLine();
                _headers = _headerLine.Split(
                    new string[] { " ", ";" }, StringSplitOptions.RemoveEmptyEntries);
                _numberOfAttributes = _headers.Length;
                string theLine;
                while ((theLine = sr.ReadLine()) != null)
                    if (theLine.Trim().Length > 4)
                        _numberOFVectors++;
            }
            double[][] result = new double[_numberOFVectors][];
            for (var w = 0; w < _numberOFVectors; w++)
            {
                result[w] = new double[_numberOfAttributes + 2]; //the two additional columns are: outlier coefficiant and vector number              
            }
            using (var sr = new StreamReader(fileName))
            {
                var theLine = sr.ReadLine();
                var v = 0;
                while ((theLine = sr.ReadLine()) != null)
                {
                    if (theLine.Trim().Length > 2)
                    {
                        string[] s = theLine.Split(
                            new string[] { " ", ";" }, StringSplitOptions.RemoveEmptyEntries);
                        var a = 0;
                        for (a = 0; a < _numberOfAttributes; a++)
                            result[v][a] = double.Parse(s[a], CultureInfo.InvariantCulture);
                        if (_headers[_headers.Length - 2].ToLower() == "outlier")
                            result[v][a] = double.Parse(s[s.Length - 2], CultureInfo.InvariantCulture);
                        else if (_headers[_headers.Length - 1].ToLower() == "outlier")
                            result[v][a] = double.Parse(s[s.Length - 1], CultureInfo.InvariantCulture);
                        else
                            result[v][a] = 1;
                        a++;
                        if (_headers[_headers.Length - 1].ToLower() == "vector")
                            result[v][a] = int.Parse(s[s.Length - 1], CultureInfo.InvariantCulture);
                        else
                            result[v][a] = v;
                        v++;
                    }
                }
            }
            _numberOfInput = result[1].Length - 3;  //the two additional columns are: outlier coefficiant and vector number
            var cl = new HashSet<int>();//cl ??
            for (int i = 0; i < result.Length; i++)
                cl.Add((int)result[i][result[1].Length - 3]);
            _numberOfOutput = cl.Count;
            _classification = false;
            if (_headerLine.ToLower().EndsWith("class") && multipleClassColumns)
            {
                _classification = true;
                var numCol = result[1].Length - 1 + cl.Count;
                double[][] dataSet = new double[result.Length][];
                for (var i = 0; i < result.Length; i++)
                    dataSet[i] = new double[numCol];
                for (var v = 0; v < result.Length; v++)
                {
                    for (var a = 0; a < result[1].Length - 3; a++)
                        dataSet[v][a] = result[v][a];
                    for (var a = result[1].Length - 2; a < result[1].Length; a++) //outlier and vector columns
                        dataSet[v][a] = result[v][a];
                    var k = (int)result[v][result[1].Length - 3]; //class column
                    var m = 0;
                    for (var a = result[1].Length - 3; a < numCol - 2; a++)
                    {
                        m++;
                        if (m == k)
                            dataSet[v][a] = 1;
                        else
                            dataSet[v][a] = transferFunction.Method.Name.Equals("SigmoidTransferFunction") ? 0 : -1;
                    }
                    dataSet[v][dataSet[0].Length - 2] = result[v][result[0].Length - 2]; //outlier
                    dataSet[v][dataSet[0].Length - 1] = result[v][result[0].Length - 1]; // v;
                }
                _data = dataSet;
            }
            else if (_headerLine.ToLower().EndsWith("class"))
            {
                _data = result;
            }
            else
            {
                _numberOfOutput = 1;
                _data = result;
            }
        }
        public int GetNumberOfHidenLayer()
        {
            return (int)Math.Sqrt(_numberOfInput * _numberOfOutput);
        }
        public int[] GetLayers()
        {
            var ll = new List<int>
            {
                _numberOfInput
            };
            ll.AddRange(new int[] { GetNumberOfHidenLayer() });
            ll.Add(_numberOfOutput);
            int[] layers = ll.ToArray();
            return layers;
        }
    }
}

﻿using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
namespace MLPProgram
{
    public class FileParser
    {
        public string HeaderLine { get; set; }
        public int NumberOfInput { get; set; }
        public int NumberOfOutput { get; set; }
        public int NumberOFVectors { get; set; }
        public int NumberOfAttributes { get; set; }
        public string[] Headers { get; set; }
        public bool Classification { get; set; }
        public double[][] Data { get; set; }
        public Func<double,double> TransferFunction{ get; set; }
        public FileParser(string fileName, Func<double, double> transferFunction, bool multipleClassColumns = true, bool firstStandardizeRun = false)
        {
            TransferFunction = transferFunction;
            GetHedersAndCountNoumbersOfVectors(fileName);
            double[][] result = new double[NumberOFVectors][];
            for (var w = 0; w < NumberOFVectors; w++)
            {
                result[w] = new double[NumberOfAttributes + 2];
                //the two additional columns are: outlier coefficiant and vector number
            }
            using (var sr = new StreamReader(fileName))
            {
                var line = sr.ReadLine();
                var v = 0;
                while ((line = sr.ReadLine()) != null)
                {
                    if (line.Trim().Length > 2)
                    {
                        string[] splitedLines = line.Split(new string[] { " ", ";" }, StringSplitOptions.RemoveEmptyEntries);
                        var a = 0;
                        for (a = 0; a < NumberOfAttributes; a++)
                            result[v][a] = double.Parse(splitedLines[a], CultureInfo.InvariantCulture);
                        if (Headers[Headers.Length - 2].ToLower() == "outlier")
                            result[v][a] = double.Parse(splitedLines[splitedLines.Length - 2], CultureInfo.InvariantCulture);
                        else if (Headers[Headers.Length - 1].ToLower() == "outlier")
                            result[v][a] = double.Parse(splitedLines[splitedLines.Length - 1], CultureInfo.InvariantCulture);
                        else
                            result[v][a] = 1;
                        a++;
                        if (Headers[Headers.Length - 1].ToLower() == "vector")
                            result[v][a] = int.Parse(splitedLines[splitedLines.Length - 1], CultureInfo.InvariantCulture);
                        else
                            result[v][a] = v;
                        v++;
                    }
                }
            }
            NumberOfInput = result[1].Length - 3;  //the two additional columns are: outlier coefficiant and vector number
            var cl = new HashSet<int>();//cl ??
            for (int i = 0; i < result.Length; i++)
                cl.Add((int)result[i][NumberOfInput]);
            NumberOfOutput = cl.Count;
            Classification = false;
            if (HeaderLine.ToLower().EndsWith("class") && multipleClassColumns)
            {
                Classification = true;
                var numCol = result[1].Length - 1 + cl.Count;
                double[][] dataSet = new double[result.Length][];
                for (var i = 0; i < result.Length; i++)
                    dataSet[i] = new double[numCol];
                for (var v = 0; v < result.Length; v++)
                {
                    for (var a = 0; a < NumberOfInput; a++)
                        dataSet[v][a] = result[v][a];
                    for (var a = result[1].Length - 2; a < result[1].Length; a++) //outlier and vector columns
                        dataSet[v][a] = result[v][a];
                    var k = (int)result[v][NumberOfInput]; //class column
                    var m = 0;
                    for (var a = NumberOfInput; a < numCol - 2; a++)
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
                Data = dataSet;
            }
            else if (HeaderLine.ToLower().EndsWith("class"))
            {
                Data = result;
            }
            else
            {
                NumberOfOutput = 1;
                Data = result;
            }
        }
        private void GetHedersAndCountNoumbersOfVectors(string fileName)
        {
            using (var sr = new StreamReader(fileName))
            {
                HeaderLine = sr.ReadLine();
                Headers = HeaderLine.Split(
                    new string[] { " ", ";" }, StringSplitOptions.RemoveEmptyEntries);
                NumberOfAttributes = Headers.Length;
                string line;
                while ((line = sr.ReadLine()) != null)
                    if (line.Trim().Length > 4)
                        NumberOFVectors++;
            }
        }
        public int GetNumberOfHidenLayer()
        {
            return (int)Math.Sqrt(NumberOfInput * NumberOfOutput);
        }
        public int[] GetLayers()
        {
            return new int[] { NumberOfInput, GetNumberOfHidenLayer(), NumberOfOutput };
        }
    }
}

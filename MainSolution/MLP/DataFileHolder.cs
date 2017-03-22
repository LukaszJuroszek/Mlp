using MLPProgram.TransferFunctions;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
namespace MLPProgram
{
    class DataFileHolder
    {
        public string HeaderLine { get; set; }
        public int NumberOfInput { get; set; }
        public int NumberOfOutput { get; set; }
        public int NumberOFVectors { get; set; }
        public int NumberOfAttributes { get; set; }
        public string[] Headers { get; set; }
        public bool Classification { get; set; }
        public double[][] Data { get; set; }
        public DataFileHolder(string fileName, ITransferFunction transferFunction, bool multipleClassColumns = true, bool firstStandardizeRun = false)
        {
            using (var sr = new StreamReader(fileName))
            {
                HeaderLine = sr.ReadLine();
                Headers = HeaderLine.Split(new string[] { " ", ";" }, StringSplitOptions.RemoveEmptyEntries);
                NumberOfAttributes = Headers.Length;
                string theLine;
                while ((theLine = sr.ReadLine()) != null)
                    if (theLine.Trim().Length > 4)
                        NumberOFVectors++;
            }
            double[][] result = new double[NumberOFVectors][];
            for (var w = 0; w < NumberOFVectors; w++)
            {
                result[w] = new double[NumberOfAttributes + 2]; //the two additional columns are: outlier coefficiant and vector number              
            }
            using (var sr = new StreamReader(fileName))
            {
                var theLine = sr.ReadLine();
                var v = 0;
                while ((theLine = sr.ReadLine()) != null)
                {
                    if (theLine.Trim().Length > 2)
                    {
                        string[] s = theLine.Split(new string[] { " ", ";" }, StringSplitOptions.RemoveEmptyEntries);
                        int a = 0;
                        for (a = 0; a < NumberOfAttributes; a++)
                            result[v][a] = Double.Parse(s[a], CultureInfo.InvariantCulture);
                        if (Headers[Headers.Length - 2].ToLower() == "outlier")
                            result[v][a] = Double.Parse(s[s.Length - 2], CultureInfo.InvariantCulture);
                        else if (Headers[Headers.Length - 1].ToLower() == "outlier")
                            result[v][a] = Double.Parse(s[s.Length - 1], CultureInfo.InvariantCulture);
                        else
                            result[v][a] = 1;
                        a++;
                        if (Headers[Headers.Length - 1].ToLower() == "vector")
                            result[v][a] = Int32.Parse(s[s.Length - 1], CultureInfo.InvariantCulture);
                        else
                            result[v][a] = v;
                        v++;
                    }
                }
            }
            NumberOfInput = result[1].Length - 3;  //the two additional columns are: outlier coefficiant and vector number
            var cl = new HashSet<int>();
            for (int i = 0; i < result.Length; i++)
                cl.Add((int)result[i][result[1].Length - 3]);
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
                            dataSet[v][a] = transferFunction is Sigmoid ? 0 : -1;
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
        public int GetNumberOfHidenLayer()
        {
            return (int)Math.Sqrt(NumberOfInput * NumberOfOutput);
        }
        public int[] GetLayers()
        {
            var ll = new List<int>
            {
                NumberOfInput
            };
            ll.AddRange(new int[] {GetNumberOfHidenLayer()});
            ll.Add(NumberOfOutput);
            int[] layers = ll.ToArray();
            return layers;
        }
    }
}

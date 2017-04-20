using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;

namespace MLPProgram
{
    public class FIleParserNew
    {
        public string HeaderLine { get; set; }
        public int NumberOfInput { get; set; }
        public int NumberOfOutput { get; set; }
        public int NumberOfInputRow { get; set; }
        public int NumberOfAttributes { get; set; }
        public string[] Headers { get; set; }
        public byte Classification { get; set; }
        public double[,] Data { get; set; }
        public Func<double, double> TransferFunction { get; set; }
        public FIleParserNew(string fileName, Func<double, double> transferFunction, bool multipleClassColumns = true, bool firstStandardizeRun = false)
        {
            TransferFunction = transferFunction;
            GetHedersAndCountNoumbersOfVectors(fileName);
            var result = new double[NumberOfInputRow, NumberOfAttributes + 2];
            //the two additional columns are: outlier coefficiant and vector number
            using (var sr = new StreamReader(fileName))
            {
                string line = sr.ReadLine(); // that will skip header line 
                int v = 0;
                while ((line = sr.ReadLine()) != null)
                {
                    if (line.Trim().Length > 2)
                    {
                        string[] splitedLines = line.Split(new string[] { " ", ";" }, StringSplitOptions.RemoveEmptyEntries);
                        var a = 0;
                        for (a = 0; a < NumberOfAttributes; a++)
                            result[v, a] = double.Parse(splitedLines[a], CultureInfo.InvariantCulture);
                        if (Headers[Headers.Length - 2].ToLower() == "outlier")
                            result[v, a] = double.Parse(splitedLines[splitedLines.Length - 2], CultureInfo.InvariantCulture);
                        else if (Headers[Headers.Length - 1].ToLower() == "outlier")
                            result[v, a] = double.Parse(splitedLines[splitedLines.Length - 1], CultureInfo.InvariantCulture);
                        else
                            result[v, a] = 1;
                        a++;
                        if (Headers[Headers.Length - 1].ToLower() == "vector")
                            result[v, a] = int.Parse(splitedLines[splitedLines.Length - 1], CultureInfo.InvariantCulture);
                        else
                            result[v, a] = v;
                        v++;
                    }
                }
            }
            NumberOfInput = result.GetLength(1) - 3;  //the two additional columns are: outlier coefficiant and vector number
            var cl = new HashSet<int>();//cl ??
            for (int i = 0; i < result.GetLength(1); i++)
                cl.Add((int)result[i, NumberOfInput]);
            NumberOfOutput = cl.Count;
            Classification = 0;
            if (HeaderLine.ToLower().EndsWith("class") && multipleClassColumns)
            {
                Classification = 1;
                int numCol = result.GetLength(1)-1 + cl.Count;
                var dataSet = new double[result.GetLength(0), numCol];
                for (int v = 0; v < result.GetLength(0); v++)
                {
                    for (int a = 0; a < NumberOfInput; a++)
                        dataSet[v, a] = result[v, a];
                    for (int a = result.Length - 2; a < result.GetLength(1); a++) //outlier and vector columns
                        dataSet[v, a] = result[v, a];
                    int k = (int)result[v, NumberOfInput]; //class column
                    int m = 0;
                    for (int a = NumberOfInput; a < numCol - 2; a++)
                    {
                        m++;
                        if (m == k)
                            dataSet[v, a] = 1;
                        else
                            dataSet[v, a] = transferFunction.Method.Name.Equals("SigmoidTransferFunction") ? 0 : -1;
                    }
                    dataSet[v, dataSet.GetLength(1) - 2] = result[v, result.GetLength(1) - 2]; //outlier
                    dataSet[v, dataSet.GetLength(1) - 1] = result[v, result.GetLength(1) - 1]; // v;
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
                Headers = HeaderLine.Split(new string[] { " ", ";" }, StringSplitOptions.RemoveEmptyEntries);
                NumberOfAttributes = Headers.Length;
                string line;
                while ((line = sr.ReadLine()) != null)
                    if (line.Trim().Length > 4)
                        NumberOfInputRow++;
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

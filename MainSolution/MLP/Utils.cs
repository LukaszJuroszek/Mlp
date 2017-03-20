using MLPProgram.TransferFunctions;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
namespace MLPProgram
{
    class Utils
    {
        public static double[][] LoadFile(string fileName,out string headerLine,out int numInput,out int numOutput,out bool classification,ITransferFunction transferFunction,bool multipleClassColumns = true,bool firstStandardizeRun = false)
        {
            var numVectors = 0;
            int numAttributes;
            string[] Headers;
            using (var sr = new StreamReader(fileName))
            {
                string theLine;
                headerLine = sr.ReadLine();
                Headers = headerLine.Split(new string[] { " ",";" },StringSplitOptions.RemoveEmptyEntries);
                numAttributes = Headers.Length;
                while (( theLine = sr.ReadLine() ) != null)
                    if (theLine.Trim().Length > 4)
                        numVectors++;
            }
            double[][] dataSet = new double[numVectors][];
            for (var w = 0;w < numVectors;w++)
            {
                dataSet[w] = new double[numAttributes + 2]; //the two additional columns are: outlier coefficiant and vector number              
            }
            using (var sr = new StreamReader(fileName))
            {
                var theLine = sr.ReadLine();
                var v = 0;
                while (( theLine = sr.ReadLine() ) != null)
                {
                    if (theLine.Trim().Length > 2)
                    {
                        string[] s = theLine.Split(new string[] { " ",";" },StringSplitOptions.RemoveEmptyEntries);
                        int a = 0;
                        for (a = 0;a < numAttributes;a++)
                            dataSet[v][a] = Double.Parse(s[a],CultureInfo.InvariantCulture);
                        if (Headers[Headers.Length - 2].ToLower() == "outlier")
                            dataSet[v][a] = Double.Parse(s[s.Length - 2],CultureInfo.InvariantCulture);
                        else if (Headers[Headers.Length - 1].ToLower() == "outlier")
                            dataSet[v][a] = Double.Parse(s[s.Length - 1],CultureInfo.InvariantCulture);
                        else
                            dataSet[v][a] = 1;
                        a++;
                        if (Headers[Headers.Length - 1].ToLower() == "vector")
                            dataSet[v][a] = Int32.Parse(s[s.Length - 1],CultureInfo.InvariantCulture);
                        else
                            dataSet[v][a] = v;
                        v++;
                    }
                }
            }
            numInput = dataSet[1].Length - 3;  //the two additional columns are: outlier coefficiant and vector number
            var cl = new HashSet<int>();
            for (int i = 0;i < dataSet.Length;i++)
                cl.Add((int)dataSet[i][dataSet[1].Length - 3]);
            numOutput = cl.Count;
            classification = false;
            if (headerLine.ToLower().EndsWith("class") && multipleClassColumns)
            {
                classification = true;
                var numCol = dataSet[1].Length - 1 + cl.Count;
                double[][] dataSet2 = new double[dataSet.Length][];
                for (var i = 0;i < dataSet.Length;i++)
                    dataSet2[i] = new double[numCol];
                for (var v = 0;v < dataSet.Length;v++)
                {
                    for (var a = 0;a < dataSet[1].Length - 3;a++)
                        dataSet2[v][a] = dataSet[v][a];
                    for (var a = dataSet[1].Length - 2;a < dataSet[1].Length;a++) //outlier and vector columns
                        dataSet2[v][a] = dataSet[v][a];
                    var k = (int)dataSet[v][dataSet[1].Length - 3]; //class column
                    var m = 0;
                    for (var a = dataSet[1].Length - 3;a < numCol - 2;a++)
                    {
                        m++;
                        if (m == k)
                            dataSet2[v][a] = 1;
                        else
                            dataSet2[v][a] = transferFunction is Sigmoid ? 0 : -1;
                    }
                    dataSet2[v][dataSet2[0].Length - 2] = dataSet[v][dataSet[0].Length - 2]; //outlier
                    dataSet2[v][dataSet2[0].Length - 1] = dataSet[v][dataSet[0].Length - 1]; // v;
                }
                return dataSet2;
            } else if (headerLine.ToLower().EndsWith("class"))
            {
                return dataSet;
            } else
            {
                numOutput = 1;
                return dataSet;
            }
        }
    }
}

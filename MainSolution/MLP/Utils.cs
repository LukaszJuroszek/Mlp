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
            double[][] DataSet = new double[numVectors][];
            for (var w = 0;w < numVectors;w++)
            {
                DataSet[w] = new double[numAttributes + 2]; //the two additional columns are: outlier coefficiant and vector number              
            }
            using (var sr = new StreamReader(fileName))
            {
                var theLine = sr.ReadLine();
                var v = 0;
                while (( theLine = sr.ReadLine() ) != null)
                {
                    if (theLine.Trim().Length > 2)
                    {
                        string[] S = theLine.Split(new string[] { " ",";" },StringSplitOptions.RemoveEmptyEntries);
                        int a = 0;
                        for (a = 0;a < numAttributes;a++)
                            DataSet[v][a] = Double.Parse(S[a],CultureInfo.InvariantCulture);
                        if (Headers[Headers.Length - 2].ToLower() == "outlier")
                            DataSet[v][a] = Double.Parse(S[S.Length - 2],CultureInfo.InvariantCulture);
                        else if (Headers[Headers.Length - 1].ToLower() == "outlier")
                            DataSet[v][a] = Double.Parse(S[S.Length - 1],CultureInfo.InvariantCulture);
                        else
                            DataSet[v][a] = 1;
                        a++;
                        if (Headers[Headers.Length - 1].ToLower() == "vector")
                            DataSet[v][a] = Int32.Parse(S[S.Length - 1],CultureInfo.InvariantCulture);
                        else
                            DataSet[v][a] = v;
                        v++;
                    }
                }
            }
            numInput = DataSet[1].Length - 3;  //the two additional columns are: outlier coefficiant and vector number
            HashSet<int> CL = new HashSet<int>();
            for (int i = 0;i < DataSet.Length;i++)
                CL.Add((int)DataSet[i][DataSet[1].Length - 3]);
            numOutput = CL.Count;
            classification = false;
            if (headerLine.ToLower().EndsWith("class") && multipleClassColumns)
            {
                classification = true;
                int numCol = DataSet[1].Length - 1 + CL.Count;
                double[][] DataSet2 = new double[DataSet.Length][];
                for (int i = 0;i < DataSet.Length;i++)
                    DataSet2[i] = new double[numCol];
                for (int v = 0;v < DataSet.Length;v++)
                {
                    for (int a = 0;a < DataSet[1].Length - 3;a++)
                        DataSet2[v][a] = DataSet[v][a];
                    for (int a = DataSet[1].Length - 2;a < DataSet[1].Length;a++) //outlier and vector columns
                        DataSet2[v][a] = DataSet[v][a];
                    int k = (int)DataSet[v][DataSet[1].Length - 3]; //class column
                    int m = 0;
                    for (int a = DataSet[1].Length - 3;a < numCol - 2;a++)
                    {
                        m++;
                        if (m == k)
                            DataSet2[v][a] = 1;
                        else
                            DataSet2[v][a] = transferFunction is Sigmoid ? 0 : -1;
                    }
                    DataSet2[v][DataSet2[0].Length - 2] = DataSet[v][DataSet[0].Length - 2]; //outlier
                    DataSet2[v][DataSet2[0].Length - 1] = DataSet[v][DataSet[0].Length - 1]; // v;
                }
                return DataSet2;
            } else if (headerLine.ToLower().EndsWith("class"))
            {
                return DataSet;
            } else
            {
                numOutput = 1;
                return DataSet;
            }
        }
    }
}

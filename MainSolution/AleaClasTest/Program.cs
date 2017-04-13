using Alea;
using Alea.CSharp;
using Alea.Parallel;
using System;
using System.Diagnostics;
using System.Linq;

namespace AleaClasTest
{
    public class Program
    {
        public SampleStruct _field;
        public Program(SampleStruct sampleStruct)
        {
            _field = sampleStruct;
        }
        static void Main(string[] args)
        {
            int size = 10;
            var arg = Enumerable.Range(1, size).ToArray();
            var arg2 = Enumerable.Range(1, size).ToArray();
            var str = new Program(new SampleStruct(size));
            var gpu = Gpu.Default;
            var array1 = gpu.Allocate(str._field.arg1);
            var array2 = gpu.Allocate(str._field.arg2);
            var result = gpu.Allocate(new double[size]);
            gpu.For(0, size, x =>
            {
                result[x] = array1[x][x][x] + array2[x][x];
                Console.WriteLine(result[x]);
            });
            var test= new double[size];
            Gpu.Copy(result, test);
            foreach (double item in test)
            {
                Console.WriteLine(item);
            }
        }
        public static void ShowArray(double[][] array)
        {
            for (int i = 0; i < array.Length; i++)
            {
                for (int j = 0; j < array.Length; j++)
                {
                    Console.Write(string.Format("{0} ", array[i][j]));
                }
                Console.Write(Environment.NewLine + Environment.NewLine);
            }
        }
    }
}
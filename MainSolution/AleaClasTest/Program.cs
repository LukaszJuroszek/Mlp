using Alea;
using Alea.CSharp;
using Alea.Parallel;
using System;
using System.Diagnostics;
using System.Linq;

namespace AleaClasTest
{
    class Program
    {
        public static void Kernel<T, TU>(Func<T, T, TU> op, TU[] result, T[] arg1, T[] arg2)
        {
            var start = blockIdx.x * blockDim.x + threadIdx.x;
            var stride = gridDim.x * blockDim.x;
            for (var i = start; i < result.Length; i += stride)
            {
                result[i] = op(arg1[i], arg2[i]);
            }
        }
        public static void KernelMlp<T>(Func<T, T, double> op, double[] tempResult, T[] arg1, T[] arg2)
        {
            var start = blockIdx.x * blockDim.x + threadIdx.x;
            var stride = gridDim.x * blockDim.x;
            for (var i = start; i < tempResult.Length; i += stride)
            {
                tempResult[i] = op(arg1[i], arg2[i]);
            }
        }
        [GpuManaged]
        static void Main(string[] args)
        {
            var size = 100000;
            var resOf= 0.0;
            var arg = Enumerable.Range(1, size).ToArray();
            var arg2 = Enumerable.Range(1, size).ToArray();
            var result = new SampleStruct<int>[size];
            var result2 = new SampleStruct<int>[size];
            var tempResult = new double[size];
            var tempResult2 = new double[size];
            Func<int, int, SampleStruct<int>> add = (x, y) => new SampleStruct<int>
            {
                add = x + y,
                multyply = x * y
            };
            Func<int, int, double> mlp = (x, y) => x * y;
            var p = Gpu.Default;
            var st = new Stopwatch();
            for (var xi = 0; xi <10; xi++)
            {
                st.Reset();
                st.Start();
                //p.Launch(Kernel, new LaunchParam(32, 512), add, result, arg, arg);
                p.Launch(KernelMlp, new LaunchParam(16, 256), mlp, tempResult, arg, arg2);
                p.Sum(tempResult);
                st.Stop();
                Console.WriteLine(st.Elapsed);
                Console.WriteLine("Gpu");
                st.Reset();
                st.Start();
                for (var i = 0; i < tempResult2.Length; i++)
                {
                    tempResult2[i] = mlp(arg[i], arg2[i]);
                }
                tempResult2.Sum();
                st.Stop();
                Console.WriteLine(st.Elapsed);
            }
            //foreach (var item in tempResult)
            //{
            //    Console.WriteLine(item);
            //}
            //foreach (var item in tempResult2)
            //{
            //    Console.WriteLine(item);
            //}

            //int numberOfSides = 5;
            //for (int i = 0; i < numberOfSides; i++)
            //{
            //    double xx =  Math.Sin(2.0 * Math.PI * i / numberOfSides);
            //    double yy =  Math.Cos(2.0 * Math.PI * i / numberOfSides);
            //    Console.WriteLine($"{xx}={yy}");
            //}

            //foreach (var item in result)
            //{
            //    Console.WriteLine($"{item.add} {item.multyply}");
            //}
        }
        public void ShowArray(int[][] array)
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
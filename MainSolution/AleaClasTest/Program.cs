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
        [GpuManaged]
        static void Main(string[] args)
        {
            //var size = 10000000;
            //var arg = Enumerable.Range(1, size).ToArray();
            //var result = new SampleStruct<int>[size];
            //var result2 = new SampleStruct<int>[size];
            //Func<int, int, SampleStruct<int>> add = (x, y) => new SampleStruct<int>
            //{
            //    add = x + y,
            //    multyply = x * y
            //};
            //var p = Gpu.Default;
            //var st = new Stopwatch();
            //for (var xi = 0; xi < 2000; xi++)
            //{
            //    st.Reset();
            //    st.Start();
            //    p.Launch(Kernel, new LaunchParam(16, 1), add, result, arg, arg);
            //         st.Stop();
            //    Console.Write(st.Elapsed.Milliseconds);
            //    st.Reset();
            //    st.Start();
            //    for (var i = 0; i < size; i++)
            //    {
            //        result2[i] = add(arg[i], arg[i]);
            //    }
            //    st.Stop();
            //    Console.Write(" ");
            //    Console.WriteLine(st.Elapsed.Milliseconds); 
            //}

            int numberOfSides = 5;
            for (int i = 0; i < numberOfSides; i++)
            {
                double xx =  Math.Sin(2.0 * Math.PI * i / numberOfSides);
                double yy =  Math.Cos(2.0 * Math.PI * i / numberOfSides);
                Console.WriteLine($"{xx}={yy}");
            }
            
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
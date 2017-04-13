using Alea;
using Alea.CSharp;
using Alea.Parallel;
using System;
using System.Diagnostics;
using System.Linq;

namespace AleaClasTest
{
     struct Program
    {
        [GpuParam] public SampleStruct _field;
        static void Main(string[] args)
        {
            int size = 2000;
            var arg = Enumerable.Range(1, size).ToArray();
            var arg2 = Enumerable.Range(1, size).ToArray();
            var str = new Program { _field = new SampleStruct(size) };
            var gpu = Gpu.Default;
            var st = new Stopwatch();
            st.Start();
            var array1 = gpu.Allocate(str._field.arg1);
            var array2 = gpu.Allocate(str._field.arg2);
            var result = gpu.Allocate(new double[size,size]);
            
            st.Stop();
            Console.WriteLine(st.Elapsed);
            st.Reset();
            st.Start();
            var test = new double[size, size];
            gpu.For(0, size, x =>
            {
                result[x,x] = array1[x,x] + array2[x,x];
                Console.WriteLine(result[x,x]);
            });
            st.Stop();
            Console.WriteLine(st.Elapsed);
            Gpu.Copy(result, test);
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
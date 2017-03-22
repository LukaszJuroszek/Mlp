using Alea;
using Alea.Parallel;
using System;
using System.Diagnostics;

namespace AleaClasTest
{
    class Program
    {
        struct Container
        {
            public int[][] _arg1;
            public int[][] _arg2;
        }
        public static int _size = 50;
        [GpuManaged]
        public double[][] Run()
        {
            var continer = new ClassWithFieldsForTest(_size);
            Container co;
            co._arg1 = continer._arg1;
            co._arg2 = continer._arg2;
            var gpu = Gpu.Default;
            var result = new double[_size][];
            for (var i = 0; i < result.Length; i++)
            {
                result[i] = new double[result.Length];
            }
            var st = new Stopwatch();
            for (var i = 0; i < _size; i++)
            {
                st.Reset();
                st.Start();
                gpu.For(0, result.Length, x =>
                {
                    for (var p = 0; p < result.Length; p++)
                        result[x][p] = co._arg1[x][p] + co._arg2[x][p] * 0.57;
                    Console.WriteLine(x);
                });
                st.Stop();
                //Console.Write(st.Elapsed);
                //Console.Write(" ");
                //st.Reset();
                //st.Start();
                //for (int s = 0; s < result.Length; s++)
                //{
                //    for (var p = 0; p < result.Length; p++)
                //        result[s][p] = co._arg1[s][p] + co._arg2[s][p];
                //}
                //st.Stop();
                //Console.WriteLine(st.Elapsed);
            }
            return result;
        }
        static void Main(string[] args)
        {
            var p = new Program();
            var x = p.Run();
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

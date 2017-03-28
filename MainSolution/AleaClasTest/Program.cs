using Alea;
using Alea.Parallel;
using System;
using System.Diagnostics;

namespace AleaClasTest
{
    class Program
    {
        [GpuParam]
        Continer continer;
        public Program(Continer continer)
        {
            this.continer = continer;
        }
        public static int _size = 50;
        [GpuManaged]
        public double[][] Run()
        {
            //co._arg1 = continer._arg1;
            //co._arg2 = continer._arg2;
            var gpu = Gpu.Default;
            var result = new double[_size][];
            for (var i = 0; i < result.Length; i++)
            {
                result[i] = new double[result.Length];
            }
            var st = new Stopwatch();
            st.Start();
            for (var i = 0; i < _size; i++)
            {
                gpu.For(1, result.Length-1, x =>
                {
                    for (var p = 0; p < result.Length; p++)
                        continer._data._results[x][p] = continer._data._arg1[x-1][p] + continer._data._arg2[x][p] * 0.57;
                    //Console.WriteLine(x);
                });
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
            st.Stop();
            Console.WriteLine(st.Elapsed.Milliseconds);
            return result;
        }
        static void Main(string[] args)
        {
            var xx = new ClassWithFieldsForTest(Program._size);
            var container = new Continer(xx);
            var p = new Program(container);
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

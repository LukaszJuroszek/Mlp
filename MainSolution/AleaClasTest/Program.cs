using Alea;
using Alea.Parallel;
using System;
using System.Diagnostics;

namespace AleaClasTest
{
    class Program
    {
        public static int _size = 50;
        [GpuParam]
        public int[][] _arg1;
        [GpuParam]
        public int[][] _arg2;
        public Program()
        {
            _arg1 = new int[_size][];
            _arg2 = new int[_size][];
            for (int i = 0; i < _size; i++)
            {
                _arg1[i] = new int[_size];
                _arg2[i] = new int[_size];

            }
            for (var i = 0; i < _arg1.GetLength(0); i++)
            {
                for (var p = 0; p < _arg1.GetLength(0); p++)
                {
                    _arg1[i][p] = p;
                    _arg2[i][p] = p;
                }
            }
        }
        [GpuManaged]
        public int[][] Run()
        {
            var gpu = Gpu.Default;
            var result = new int[_size][];
            for (var i = 0; i < result.Length; i++)
            {
                result[i] = new int[result.Length];
            }
            var st = new Stopwatch();
            for (var i = 0; i < _size; i++)
            {
                st.Reset();
                st.Start();
                gpu.For(0, result.Length, x =>
                {
                    for (var p = 0; p < result.Length; p++)
                        result[x][p] = _arg1[x][p] + _arg2[x][p];
                });
                st.Stop();
                Console.WriteLine(st.Elapsed);
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

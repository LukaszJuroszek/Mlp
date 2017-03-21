using Alea;
using Alea.Parallel;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AleaClasTest
{
    class Program
    {
        public static int _length = 50000000;
        public int[] _arg1;
        public int[] _arg2;
        public Program()
        {
            _arg1 = Enumerable.Range(0, _length).ToArray();
            _arg2 = Enumerable.Range(0, _length).ToArray();
        }
        //[GpuManaged]
        public void Run()
        {
            var gpu = Gpu.Default;
            var result = new int[_length];
            var st = new Stopwatch();
            for (var i = 0; i < 30; i++)
            {
                st.Start();
                gpu.For(0, result.Length, x => result[x] = _arg1[x] + _arg2[x]);
                st.Stop();
                Console.WriteLine(st.Elapsed);
            }
        }
        [GpuManaged]
        static void Main(string[] args)
        {
            var p = new Program();
            p.Run();
        }
    }
}

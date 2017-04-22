using Alea;
using Alea.CSharp;
using Alea.Parallel;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace AleaClasTest
{
    struct ByteTestInner
    {
        //Func<int, double> _op;
        //public ByteTestInner(Func<int, double> opp)
        //{
        //    _op = opp;
        //}
        byte[] _byteValue;
        public ByteTestInner(byte byteValue)
        {
            _byteValue = new byte[2];
            for (int i = 0; i < 2; i++)
            {
                _byteValue[i] = byteValue;
            }
        }
    }
    struct ByteTest
    {
        byte _byteValue;
        ByteTestInner _inner;
        public ByteTest(byte byteValue, ByteTestInner inner)
        {
            _byteValue = byteValue;
            _inner = inner;
        }
    }
    struct BoolTest
    {
        bool _boolValue;
        public BoolTest(bool boolValue)
        {
            _boolValue = boolValue;
        }
    }
    public enum TestEnum
    {
        One = 0,
        Second = 1
    }
    struct Program
    {
        [GpuParam] public SampleStruct _field;
        static void Main(string[] args)
        {
            Func<int, double> op = (x) => 1;
            var testOfBools = new BoolTest[2];
            var testOfBytes = new ByteTest[2];
            testOfBools[0] = new BoolTest(true);
            testOfBools[1] = new BoolTest(false);
            testOfBytes[0] = new ByteTest(0, new ByteTestInner(1));
            testOfBytes[1] = new ByteTest(1, new ByteTestInner(1));
            int size = 2000;
            var arg = Enumerable.Range(1, size).ToArray();
            var arg2 = Enumerable.Range(1, size).ToArray();
            var str = new Program { _field = new SampleStruct(size) };
            var gpu = Gpu.Default;
            //throws Excp var allocatedBool = gpu.Allocate(testOfBools);
            var st = new Stopwatch();
            st.Start();
            var testEnu = new Dictionary<TestEnum, double[,]>
            {
                { TestEnum.One, str._field.arg1 },
                { TestEnum.Second, str._field.arg2 }
            };
            var array1 = gpu.Allocate(testEnu.ElementAt(0).Value);
            var array2 = gpu.Allocate(str._field.arg2);
            var result = gpu.Allocate(new double[size, size]);
            var teste = new double[3][,];
            teste[0] = str._field.arg1;
            teste[1] = str._field.arg2;
            teste[2] = str._field.arg2;
            var results = gpu.Allocate(teste);
            st.Stop();
            Console.WriteLine(st.Elapsed);
            st.Reset();
            st.Start();
            var test = new double[size, size];
            gpu.For(0, size, x =>
            {
                result[x, x] = teste[0][1, 1] + array2[x, x];
                Console.WriteLine(result[x, x]);
            });
            st.Stop();
            Console.WriteLine(st.Elapsed);
            Gpu.Copy(result, test);
            var temp = new double[3][,];
            teste[0] = str._field.arg1;
            teste[1] = str._field.arg2;
            teste[2] = str._field.arg2;


            //Gpu.Copy(results, temp); //throws error 

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
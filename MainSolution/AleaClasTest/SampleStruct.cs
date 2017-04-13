using System.Linq;

namespace AleaClasTest
{
    public struct SampleStruct
    {
        public double[][][] arg1;
        public double[][] arg2;
        public SampleStruct(int size)
        {
            arg1 = new double[size][][];
            arg2 = new double[size][];
            for (int i = 0; i < size; i++)
            {
                arg1[i] = new double[size][];
                arg2[i] = new double[size];
                for (int p = 0; p < size; p++)
                {
                    arg2[i][p] = p;
                    arg1[i][p] = new double[size];
                    for (int x = 0; x < size; x++)
                    {
                        arg1[i][p][x] = x;

                    }
                }
            }
        }
    }

}
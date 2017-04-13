using System.Linq;

namespace AleaClasTest
{
    public struct SampleStruct
    {
        public double[,] arg1;
        public double[,] arg2;
        public SampleStruct(int size)
        {
            arg1 = new double[size,size];
            arg2 = new double[size,size];
            for (int i = 0; i < size; i++)
            {
                    arg1[i,i] = i;
                    arg2[i,i] = i;
            }
        }
    }

}
namespace AleaClasTest
{
    public struct ClassWithFieldsForTest
    {
        public int[][] _arg1;
        public int[][] _arg2;
        public double[][] _results;
        public ClassWithFieldsForTest(int size)
        {
            _arg1 = new int[size][];
            _arg2 = new int[size][];
            _results = new double[size][];
            for (var i = 0; i < size; i++)
            {
                _arg1[i] = new int[size];
                _arg2[i] = new int[size];
                _results[i] = new double[size];
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
    }
}
using Alea;

namespace AleaClasTest
{
    public struct Continer
    {
        [GpuParam]
        public ClassWithFieldsForTest _data;
        public Continer(ClassWithFieldsForTest data)
        {
            _data = data;
        }
    }
}

using System;

namespace MLPProgram.Networks
{
    interface INetwork
    {
        double Accuracy(out double mse, int lok=0);
    }
}

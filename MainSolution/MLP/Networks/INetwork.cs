using System;

namespace MLPProgram.Networks
{
    interface INetwork
    {
        double Accuracy(double[][] DataSet, out double mse, Func<double, double> transferFunction, int lok=0);
    }
}

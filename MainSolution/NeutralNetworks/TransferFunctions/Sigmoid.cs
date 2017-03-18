using System;

namespace DL.TransferFunctions
{
    class Sigmoid : ITransferFunction
    {
        public double TransferFunction(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        public double Derivative(double x)
        {
            return x * (1.0 - x);
        }
    }
}

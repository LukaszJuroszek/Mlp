﻿using System;
namespace MLPProgram.TransferFunctions
{
    class HyperbolicTangent : ITransferFunction
    {
        public double TransferFunction(double x)
        {
            return Math.Tanh(x);
        }
        public double Derivative(double x)
        {
            return 1.0 - x * x; 
        }
    }
}

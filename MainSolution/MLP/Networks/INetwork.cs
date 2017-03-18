using MLPProgram.TransferFunctions;

namespace MLPProgram.Networks
{
    interface INetwork
    {
        double Accuracy(double[][] DataSet, out double mse, ITransferFunction transferFunction, int lok=0);
    }
}

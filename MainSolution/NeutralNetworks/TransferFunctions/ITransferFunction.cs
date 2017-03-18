namespace NeutralNetworks.TransferFunctions
{
    interface ITransferFunction
    {
        double TransferFunction(double x);
        double Derivative(double x);
    }
}

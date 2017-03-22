namespace MLPProgram.TransferFunctions
{
   public interface ITransferFunction
    {
        double TransferFunction(double x);
        double Derivative(double x);
    }
}

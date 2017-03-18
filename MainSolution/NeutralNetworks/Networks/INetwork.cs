namespace NeutralNetworks.Networks
{
    interface INetwork
    {
        double Accuracy(double[][] DataSet, out double mse, TransferFunctions.ITransferFunction transferFunction, int lok=0);
        double[] GetSignalErrorTable(double[][] DataSet, ref double accuracy, DataSelection.IDataSelection dataSelection, double errorExponent = 2.0);       
    }
}

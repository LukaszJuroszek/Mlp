using MLPProgram;
using MLPProgram.LearningAlgorithms;
using MLPProgram.Networks;
using System;
using System.Diagnostics;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace MLPTests
{
    public class Tests
    {
        private readonly string _filePath = @"..\..\Datasets\ionosphere_std_sh.txt";
        private readonly double _deltaValue = 0.1;
        private readonly double _errorExponent = 2.0;
        private readonly ITestOutputHelper _outputHelper;
        public Tests(ITestOutputHelper testOutputHelper)
        {
            _outputHelper = testOutputHelper;
        }
        [Fact]
        public void IsTwoFileParserAreEq()
        {
            var fileParser = new FileParser(_filePath, GradientLearning.SigmoidTransferFunction);
            var fileParserNew = new FIleParserNew(_filePath, GradientLearning.SigmoidTransferFunction);
            Assert.Equal(fileParser.Classification, fileParserNew.Classification);
            Assert.Equal(fileParser.HeaderLine, fileParserNew.HeaderLine);
            Assert.Equal(fileParser.NumberOfAttributes, fileParserNew.NumberOfAttributes);
            Assert.Equal(fileParser.NumberOfInput, fileParserNew.NumberOfInput);
            Assert.Equal(fileParser.NumberOfInputRow, fileParserNew.NumberOfInputRow);
            Assert.Equal(fileParser.NumberOfOutput, fileParserNew.NumberOfOutput);
            Assert.Equal(fileParser.Headers, fileParserNew.Headers);
            Assert.Equal(fileParser.TransferFunction, fileParserNew.TransferFunction);
            Assert.Equal(fileParser.Data.GetLength(0), fileParserNew.Data.GetLength(0));
            Assert.Equal(fileParser.Data[0].GetLength(0), fileParserNew.Data.GetLength(1));
            Assert.NotNull(fileParser.Data);
            Assert.NotNull(fileParserNew.Data);
            for (int i = 0; i < fileParser.Data.GetLength(0); i++)
                for (int n = 0; n < fileParser.Data[i].GetLength(0); n++)
                    Assert.Equal(fileParser.Data[i][n], fileParserNew.Data[i, n]);
        }
        [Fact]
        public void IsTwoDataHoldersAreEq()
        {
            var fileParser = new FileParser(_filePath, GradientLearning.SigmoidTransferFunction);
            var fileParserNew = new FIleParserNew(_filePath, GradientLearning.SigmoidTransferFunction);
            var data = new BaseDataHolder(fileParser);
            var dataNew = new DataHolder(fileParserNew);
            Assert.Equal(data._classification, dataNew._classification);
            Assert.Equal(data._isSigmoidFunction, dataNew._isSigmoidFunction);
            Assert.Equal(data._layer, dataNew._layer);
            Assert.Equal(data._numberOfInput, dataNew._numberOfInput);
            Assert.Equal(data._numberOfOutput, dataNew._numberOfOutput);
            Assert.Equal(data._numberOFVectors, dataNew._numberOfInputRow);
            Assert.NotNull(data._trainingDataSet);
            Assert.NotNull(dataNew._trainingDataSet);
            for (int i = 0; i < data._trainingDataSet.GetLength(0); i++)
                for (int n = 0; n < data._trainingDataSet[i].GetLength(0); n++)
                    Assert.Equal(data._trainingDataSet[i][n], dataNew._trainingDataSet[i, n]);
        }
        [Fact]
        public void IsTwoMLPNetworksAreEq()
        {
            var fileParser = new FileParser(_filePath, GradientLearning.SigmoidTransferFunction);
            var fileParserNew = new FIleParserNew(_filePath, GradientLearning.SigmoidTransferFunction);
            var data = new BaseDataHolder(fileParser);
            var dataNew = new DataHolder(fileParserNew);
            var net = new MLP(data);
            var netNew = new MLPNew(dataNew, net.weights);
            for (int l = 1; l < net.delta.GetLength(0); l++)
                for (int n = 0; n < net.delta[l].GetLength(0); n++)
                    for (int w = 0; w < net.delta[l][n].GetLength(0); w++)
                    {
                        Assert.Equal(net.delta[l][n][w], netNew.delta[l][n, w], 6);
                        Assert.Equal(net.weightDiff[l][n][w], netNew.weightDiff[l][n, w], 6);
                        Assert.Equal(net.weights[l][n][w], netNew.weights[l][n, w], 6);
                        Assert.Equal(net.prevWeightDiff[l][n][w], netNew.prevWeightDiff[l][n, w], 6);
                    }
            for (int l = 0; l < 1; l++)
                for (int n = 0; n < net.output[l].GetLength(0); n++)
                    Assert.Equal(net.output[l][n], netNew.output[l][n]);
            for (int l = 1; l < net.signalError.GetLength(0); l++)
                for (int n = 0; n < net.signalError[l].GetLength(0); n++)
                {
                    Assert.Equal(net.signalError[l][n], netNew.signalError[l][n]);
                    Assert.Equal(net.output[l][n], netNew.output[l][n]);
                }
            for (int i = 0; i < net.layer.GetLength(0); i++)
                Assert.Equal(net.layer[i], netNew.networkLayers[i]);
            Assert.Equal(net.numbersOfLayers, netNew.numbersOfLayers);
        }
        [Fact]
        public void AreForwardPassWorkingInTheSameWay()
        {
            var fileParser = new FileParser(_filePath, GradientLearning.SigmoidTransferFunction);
            var fileParserNew = new FIleParserNew(_filePath, GradientLearning.SigmoidTransferFunction);
            var data = new BaseDataHolder(fileParser);
            var dataNew = new DataHolder(fileParserNew);
            var net = new MLP(data);
            var netNew = new MLPNew(dataNew, net.weights);
            var netNewSecond = new MLPNew(dataNew, net.weights);
            for (int v = 0; v < net.baseData._trainingDataSet.Length; v++)
            {
                Program.ForwardPass(net, v);
                Program.ForwardPass(netNew, v);
                Program.ForwardPass(
                    netNewSecond.weights,
                    netNewSecond.networkLayers,
                    netNewSecond.output,
                    netNewSecond.baseData._trainingDataSet,
                    netNewSecond.numbersOfLayers,
                    netNewSecond.classification,
                    netNewSecond.baseData._isSigmoidFunction,
                    v);
            }
            IsTwoMLPNetworksAreEq(net, netNew);
            IsTwoMLPNetworksAreEq(net, netNewSecond);
        }
        [Fact]
        public void AreCreateWeightZeroAndAsingDeltaValueWorkingInMLPs()
        {
            var fileParser = new FileParser(_filePath, GradientLearning.SigmoidTransferFunction);
            var fileParserNew = new FIleParserNew(_filePath, GradientLearning.SigmoidTransferFunction);
            var data = new BaseDataHolder(fileParser);
            var dataNew = new DataHolder(fileParserNew);
            var net = new MLP(data);
            var netNew = new MLPNew(dataNew, net.weights);
            GradientLearning.CreateWeightZeroAndAsingDeltaValue(net, _deltaValue);
            TrainingSystem.CreateWeightZeroAndAsingDeltaValue(netNew, _deltaValue);
            IsTwoMLPNetworksAreEq(net, netNew);

        }
        [Fact]
        public void AreCalculateSignalErrorForOutputLayerWorkingInMLPs()
        {
            var fileParser = new FileParser(_filePath, GradientLearning.SigmoidTransferFunction);
            var fileParserNew = new FIleParserNew(_filePath, GradientLearning.SigmoidTransferFunction);
            var data = new BaseDataHolder(fileParser);
            var dataNew = new DataHolder(fileParserNew);
            var net = new MLP(data);
            var netNew = new MLPNew(dataNew, net.weights);

            TrainingSystem.CreateWeightZeroAndAsingDeltaValue(netNew, _deltaValue);
            GradientLearning.CreateWeightZeroAndAsingDeltaValue(net, _deltaValue);
            for (int batch = 0; batch < net.baseData._numberOFVectors; batch++)
            {
                TrainingSystem.MakeGradientZero(netNew);
                GradientLearning.MakeGradientZero(net);
                Program.ForwardPass(netNew, batch);//TrainingSystem
                Program.ForwardPass(net, batch);//GradientLearning  
                for (int l = 0; l < net.baseData._numberOfOutput; l++)
                {
                    double expected = GradientLearning.CalculateSignalErrorsForOutputLayer(net, batch, l, _errorExponent);
                    double actual = TrainingSystem.CalculateSignalErrorsForOutputLayer(netNew, batch, l, _errorExponent);
                    net.signalError[net.numbersOfLayers - 1][l] = expected;
                    netNew.signalError[(int)NetworkLayer.Output][l] = actual;
                    Assert.Equal(expected, actual);
                }
                IsTwoMLPNetworksAreEq(net, netNew);
            }
        }
        public  void IsTwoMLPNetworksAreEq(MLP net, MLPNew netNew)
        {
          
            for (int l = 1; l < net.delta.GetLength(0); l++)
                for (int n = 0; n < net.delta[l].GetLength(0); n++)
                    for (int w = 0; w < net.delta[l][n].GetLength(0); w++)
                    {
                        Assert.Equal(net.delta[l][n][w], netNew.delta[l][n, w]);
                        Assert.Equal(net.weightDiff[l][n][w], netNew.weightDiff[l][n, w], 6);
                        Assert.Equal(net.weights[l][n][w], netNew.weights[l][n, w], 6);
                        Assert.Equal(net.prevWeightDiff[l][n][w], netNew.prevWeightDiff[l][n, w], 6);
                    }
            for (int l = 0; l < 1; l++)
                for (int n = 0; n < net.output[l].GetLength(0); n++)
                    Assert.Equal(net.output[l][n], netNew.output[l][n]);
            for (int l = 1; l < net.signalError.GetLength(0); l++)
                for (int n = 0; n < net.signalError[l].GetLength(0); n++)
                {
                    Assert.Equal(net.signalError[l][n], netNew.signalError[l][n]);
                    Assert.Equal(net.output[l][n], netNew.output[l][n]);
                }
            for (int i = 0; i < net.layer.GetLength(0); i++)
                Assert.Equal(net.layer[i], netNew.networkLayers[i]);
            Assert.Equal(net.numbersOfLayers, netNew.numbersOfLayers);
        }
    }
}

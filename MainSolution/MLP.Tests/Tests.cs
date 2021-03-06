﻿using MLPProgram;
using MLPProgram.LearningAlgorithms;
using MLPProgram.Networks;
using Xunit;
using Xunit.Abstractions;

namespace MLPTests
{
    public class Tests
    {
        private readonly string _filePath = @"..\..\Datasets\ionosphere_std_sh.txt";
        private readonly double _deltaValue = 0.1;
        private readonly double _learnRate = 0.05;
        private readonly double _momentum = 0.5;
        private readonly double _errorExponent = 2.0;
        private readonly double _etaPlus = 1.2;
        private readonly double _etaMinus = 0.5;
        private readonly double _minDelta = 0.00001;
        private readonly double _maxDelta = 10;
        private readonly int _numberOfEpochos = 50;
        private readonly int _accu = 7;
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
            Assert.Equal(fileParser.Classification, fileParserNew.Classification == 1);
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
            Assert.Equal(data._classification, dataNew._classification == 1);
            Assert.Equal(data._isSigmoidFunction, dataNew._isSigmoidFunction == 1);
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
                        Assert.Equal(net.delta[l][n][w], netNew.delta[l][n, w], _accu);
                        Assert.Equal(net.weightDiff[l][n][w], netNew.weightDiff[l][n, w], _accu);
                        Assert.Equal(net.weights[l][n][w], netNew.weights[l][n, w], _accu);
                        Assert.Equal(net.prevWeightDiff[l][n][w], netNew.prevWeightDiff[l][n, w], _accu);
                    }
            for (int l = 0; l < 1; l++)
                for (int n = 0; n < net.output[l].GetLength(0); n++)
                    Assert.Equal(net.output[l][n], netNew.output[l][n]);
            for (int l = 1; l < net.signalError.GetLength(0); l++)
                for (int n = 0; n < net.signalError[l].GetLength(0); n++)
                {
                    Assert.Equal(net.signalError[l][n], netNew.signalError[l][n], _accu);
                    Assert.Equal(net.output[l][n], netNew.output[l][n], _accu);
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
            var netNewForGpu = new MLPNew(dataNew, net.weights);
            for (int v = 0; v < net.baseData._trainingDataSet.Length; v++)
            {
                Program.ForwardPass(net, v);
                Program.ForwardPass(netNew, v);
                Program.ForwardPassGpu(netNewForGpu, v);
                Program.ForwardPass(netNewSecond.weights,
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
            IsTwoMLPNetworksAreEq(net, netNewForGpu);
        }
        [Fact]
        public void IsTrainingMethodWorkInMLPsOldStyle()
        {
            var fileParser = new FileParser(_filePath, GradientLearning.SigmoidTransferFunction);
            var fileParserNew = new FIleParserNew(_filePath, GradientLearning.SigmoidTransferFunction);
            var data = new BaseDataHolder(fileParser);
            var dataNew = new DataHolder(fileParserNew);
            var net = new MLP(data);
            var netNew = new MLPNew(dataNew, net.weights);
            var netForGpu = new MLPNew(dataNew, net.weights);
            AssertThatCreateWeigthZeroAndAsignDeltaValueWorkInTheSameWayInMLPs(net, netNew, netForGpu);
            for (int epoch = 0; epoch < _numberOfEpochos; epoch++)
            {
                ActThatMakeZeroGradientWorkingInMLPs(net, netNew, netForGpu);
                for (int batch = 0; batch < net.baseData._numberOFVectors; batch++)
                {
                    ActForwardPassWorkingInMLPs(net, netNew, netForGpu, batch);
                    AssertSignalErrosInOutputLayer(net, netNew, netForGpu, batch);
                    AssertSignalErrorsInHiddenLayer(net, netNew, netForGpu);
                    ActBiases(net, netNew, netForGpu);
                }
                ActUpgradeWeithsWorkingInMLPs(net, netNew, netForGpu);
                ActThatMakeZeroGradientWorkingInMLPs(net, netNew, netForGpu);
                IsTwoMLPNetworksAreEq(net, netNew);
                IsTwoMLPNetworksAreEq(net, netForGpu);
            }
            AssertOfAccuracy(net, netNew, netForGpu);
            IsTwoMLPNetworksAreEq(net, netNew);
            IsTwoMLPNetworksAreEq(net, netForGpu);

        }
        [Fact]
        public void IsTrainingMethodWorkInMLPsNewStyle()
        {
            var fileParser = new FileParser(_filePath, GradientLearning.SigmoidTransferFunction);
            var fileParserNew = new FIleParserNew(_filePath, GradientLearning.SigmoidTransferFunction);
            var data = new BaseDataHolder(fileParser);
            var dataNew = new DataHolder(fileParserNew);
            var net = new MLP(data);
            var netNew = new MLPNew(dataNew, net.weights);
            var netForGpu = new MLPNew(dataNew, net.weights);

            AssertThatCreateWeigthZeroAndAsignDeltaValueWorkInTheSameWayInMLPs(net, netNew, netForGpu);
            for (int epoch = 0; epoch < _numberOfEpochos; epoch++)
            {
                ActThatMakeZeroGradientWorkingInMLPs(net, netNew, netForGpu);
                ActCalculateForRowsAndVectorWorkInMLPs(net, netNew, netForGpu);
                ActUpgradeWeithsWorkingInMLPs(net, netNew, netForGpu);
                ActThatMakeZeroGradientWorkingInMLPs(net, netNew, netForGpu);
                IsTwoMLPNetworksAreEq(net, netNew);
                IsTwoMLPNetworksAreEq(net, netForGpu);
            }
            AssertOfAccuracy(net, netNew, netForGpu);
            IsTwoMLPNetworksAreEq(net, netNew);
            IsTwoMLPNetworksAreEq(net, netForGpu);
        }
        private void ActCalculateForRowsAndVectorWorkInMLPs(MLP net, MLPNew netNew, MLPNew netForGpu)
        {
            GradientLearning.CalculateForAllVectors(net, _errorExponent, _learnRate);
            TrainingSystem.CalculateForAllRow(netNew, _errorExponent, _learnRate);
            TrainingSystemGpu.CalculateForAllRowGpu(netForGpu, _errorExponent, _learnRate);
        }
        public void AssertOfAccuracy(MLP net, MLPNew netNew, MLPNew netForGpu)
        {
            double result = MLP.CountAccuracy(net);
            double resultNew = MLPNew.CountAccuracy(netNew);
            double resultForGpu = MLPNew.CountAccuracy(netForGpu);
            Assert.Equal(result, resultNew, _accu);
            Assert.Equal(result, resultForGpu, _accu);

        }
        private void ActUpgradeWeithsWorkingInMLPs(MLP net, MLPNew netNew, MLPNew netForGpu)
        {
            GradientLearning.UpdateWeightsRprop(net, _learnRate, _momentum, _etaPlus, _etaMinus, _minDelta, _maxDelta);
            TrainingSystem.UpdateWeightsRprop(netNew, _learnRate, _momentum, _etaPlus, _etaMinus, _minDelta, _maxDelta);
            TrainingSystemGpu.UpdateWeightsRpropGpu(netForGpu, _learnRate, _momentum, _etaPlus, _etaMinus, _minDelta, _maxDelta);
        }
        private void ActForwardPassWorkingInMLPs(MLP net, MLPNew netNew, MLPNew netForGpu, int batch)
        {
            Program.ForwardPass(net, batch);//GradientLearning
            Program.ForwardPass(netNew, batch);//TrainingSystem
            Program.ForwardPassGpu(netForGpu, batch);//TrainingSystemGpu
        }
        private void ActThatMakeZeroGradientWorkingInMLPs(MLP net, MLPNew netNew, MLPNew netForGpu)
        {
            GradientLearning.MakeGradientZero(net);
            TrainingSystem.MakeGradientZero(netNew);
            TrainingSystemGpu.MakeGradientZeroGpu(netForGpu);
        }
        private void ActBiases(MLP net, MLPNew netNew, MLPNew netForGpu)
        {
            GradientLearning.CalculateBias(net, _learnRate);
            TrainingSystem.CalculateBias(netNew, _learnRate);
            TrainingSystemGpu.CalculateBiasOnGpu(netForGpu, _learnRate);
        }
        private void AssertThatCreateWeigthZeroAndAsignDeltaValueWorkInTheSameWayInMLPs(MLP net, MLPNew netNew, MLPNew netForGpu)
        {
            GradientLearning.CreateWeightZeroAndAsingDeltaValue(net, _deltaValue);
            TrainingSystem.CreateWeightZeroAndAsingDeltaValue(netNew, _deltaValue);
            TrainingSystemGpu.CreateWeightZeroAndAsingDeltaValueGpu(netForGpu, _deltaValue);
            for (int l = 1; l < netNew.numbersOfLayers; l++)
                for (int n = 0; n < netNew.networkLayers[l]; n++)
                    for (int w = 0; w <= netNew.networkLayers[l - 1]; w++)
                    {
                        Assert.Equal(net.weightDiff[l][n][w], netNew.weightDiff[l][n, w]);
                        Assert.Equal(net.weightDiff[l][n][w], netForGpu.weightDiff[l][n, w]);
                        Assert.Equal(net.delta[l][n][w], netNew.delta[l][n, w]);
                        Assert.Equal(net.delta[l][n][w], netForGpu.delta[l][n, w]);
                    }
        }
        private void AssertSignalErrorsInHiddenLayer(MLP net, MLPNew netNew, MLPNew netForGpu)
        {
            for (int l = net.numbersOfLayers - 2; l > 0; l--)
                for (int n = 0; n < net.layer[l]; n++)
                {
                    double expected = GradientLearning.CalculateSignalErrorFroHiddenLayer(net, l, n);
                    double actual = TrainingSystem.CalculateSignalErrorFroHiddenLayer(netNew, l, n);
                    double actualGpu = TrainingSystemGpu.CalculateSignalErrorFroHiddenLayerGpu(netForGpu, l, n);
                    net.signalError[l][n] = expected;
                    netNew.signalError[l][n] = actual;
                    netForGpu.signalError[l][n] = actualGpu;
                    Assert.Equal(expected, actual, _accu);
                    Assert.Equal(expected, actualGpu, _accu);
                }
        }
        private void AssertSignalErrosInOutputLayer(MLP net, MLPNew netNew, MLPNew netForGpu, int batch)
        {
            for (int l = 0; l < net.baseData._numberOfOutput; l++)
            {
                double expected = GradientLearning.CalculateSignalErrorsForOutputLayer(net, batch, l, _errorExponent);
                double actual = TrainingSystem.CalculateSignalErrorsForOutputLayer(netNew, batch, l, _errorExponent);
                double actualGpu = TrainingSystemGpu.CalculateSignalErrorsForOutputLayerGpu(netForGpu, batch, l, _errorExponent);
                net.signalError[net.numbersOfLayers - 1][l] = expected;
                netNew.signalError[(int)NetworkLayer.Output][l] = actual;
                netForGpu.signalError[(int)NetworkLayer.Output][l] = actualGpu;
                Assert.Equal(expected, actual, _accu);
                Assert.Equal(expected, actualGpu, _accu);
            }
        }
        public void IsTwoMLPNetworksAreEq(MLP net, MLPNew netNew)
        {
            for (int l = 1; l < net.delta.GetLength(0); l++)
                for (int n = 0; n < net.delta[l].GetLength(0); n++)
                    for (int w = 0; w < net.delta[l][n].GetLength(0); w++)
                    {
                        Assert.Equal(net.delta[l][n][w], netNew.delta[l][n, w], _accu);
                        Assert.Equal(net.weightDiff[l][n][w], netNew.weightDiff[l][n, w], _accu);
                        Assert.Equal(net.weights[l][n][w], netNew.weights[l][n, w], _accu);
                        Assert.Equal(net.prevWeightDiff[l][n][w], netNew.prevWeightDiff[l][n, w], _accu);
                    }
            for (int l = 0; l < 1; l++)
                for (int n = 0; n < net.output[l].GetLength(0); n++)
                    Assert.Equal(net.output[l][n], netNew.output[l][n], _accu);
            for (int l = 1; l < net.signalError.GetLength(0); l++)
                for (int n = 0; n < net.signalError[l].GetLength(0); n++)
                {
                    Assert.Equal(net.signalError[l][n], netNew.signalError[l][n], _accu);
                    Assert.Equal(net.output[l][n], netNew.output[l][n], _accu);
                }
            for (int i = 0; i < net.layer.GetLength(0); i++)
                Assert.Equal(net.layer[i], netNew.networkLayers[i]);
            Assert.Equal(net.numbersOfLayers, netNew.numbersOfLayers);
        }
    }
}

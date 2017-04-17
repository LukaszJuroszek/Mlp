using MLPProgram;
using MLPProgram.LearningAlgorithms;
using MLPProgram.Networks;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace MLPTests
{
    public class Tests
    {
        string _filePath = @"..\..\Datasets\ionosphere_std_sh.txt";
        [Fact]
        public void IsTwoFileParserAreEq()
        {
            //arrange, act
            var fileParser = new FileParser(_filePath, GradientLearning.SigmoidTransferFunction);
            var fileParserNew = new FIleParserNew(_filePath, GradientLearning.SigmoidTransferFunction);
            //asert
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
            //arrange 
            var fileParser = new FileParser(_filePath, GradientLearning.SigmoidTransferFunction);
            var fileParserNew = new FIleParserNew(_filePath, GradientLearning.SigmoidTransferFunction);
            //act
            var data = new BaseDataHolder(fileParser);
            var dataNew = new DataHolder(fileParserNew);
            //asert
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
            //arrange 
            var fileParser = new FileParser(_filePath, GradientLearning.SigmoidTransferFunction);
            var fileParserNew = new FIleParserNew(_filePath, GradientLearning.SigmoidTransferFunction);
            var data = new BaseDataHolder(fileParser);
            var dataNew = new DataHolder(fileParserNew);
            //act
            var net = new MLP(data);
            var netNew = new MLPNew(dataNew, net.weights);
            for (int l = 1 ; l < net.delta.GetLength(0); l++)
                for (int n = 0; n < net.delta[l].GetLength(0); n++)
                    for (int w = 0; w < net.delta[l][n].GetLength(0); w++)
                    {
                        Assert.Equal(net.delta[l][n][w], netNew.delta[l][n, w],6);
                        Assert.Equal(net.weightDiff[l][n][w], netNew.weightDiff[l][n, w],6);
                        Assert.Equal(net.weights[l][n][w], netNew.weights[l][n, w],6);
                        Assert.Equal(net.prevWeightDiff[l][n][w], netNew.prevWeightDiff[l][n, w],6);
                    }
            for (int l =0; l < 1; l++)
                for (int n = 0; n < net.output[l].GetLength(0); n++)
                    Assert.Equal(net.output[l][n], netNew.output[l][n]);
            for (int l =1;l < net.signalError.GetLength(0); l++)
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
        public void IsTwoLearningMethodsCountsInTheSameWay()
        {
            //arrange 
            var fileParser = new FileParser(_filePath, GradientLearning.SigmoidTransferFunction);
            var fileParserNew = new FIleParserNew(_filePath, GradientLearning.SigmoidTransferFunction);
            var data = new BaseDataHolder(fileParser);
            var dataNew = new DataHolder(fileParserNew);
            var net = new MLP(data);
            var netNew = new MLPNew(dataNew, net.weights);
            var learningAlgorithm = new GradientLearning(net);
            var trainingSystems = new TrainingSystem(netNew);

        }
    }
}

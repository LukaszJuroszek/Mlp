using System;
using System.Collections.Generic;
using System.Linq;

namespace MLPProgram.Networks
{
    public enum NetworkLayer
    {
        Input = 0,
        Hidden = 1,
        Output = 2
    }
    public struct MLPNew
    {
        public Dictionary<NetworkLayer, double[,]> weightDiff, prevWeightDiff, delta, weights;
        public Dictionary<NetworkLayer, double[]> signalError, output;
        public int[] networkLayers;
        public int numbersOfLayers;
        public bool classification;
        public DataHolder baseData;
        public MLPNew(DataHolder data)
        {
            var rnd = new Random();
            baseData = data;
            classification = data._classification;
            networkLayers = data._layer;
            numbersOfLayers = networkLayers.Length;
            weights = Create2DLayers(networkLayers);
            weightDiff = Create2DLayers(networkLayers);
            prevWeightDiff = Create2DLayers(networkLayers);
            delta = Create2DLayers(networkLayers);
            signalError = Create1DLayers(networkLayers);
            output = CreateFull1DLayers(networkLayers);
            double dw0 = 0.20;
            for (int l = 0; l< numbersOfLayers; l++)
            {
                var weightsItem = weights.ElementAt(l);
                var deltaItem = delta.ElementAt(l);
                var itemKey = weightsItem.Key;
                if (itemKey == NetworkLayer.Output || itemKey == NetworkLayer.Hidden)
                {
                var weightsItemValue = weightsItem.Value;
                var deltaItemValue = deltaItem.Value;
                    for (int n = 0; n < weightsItemValue.GetLength(0); n++)
                    {
                        for (int w = 0; w < weightsItemValue.GetLength(1); w++)
                        {
                            weightsItemValue[n,w]= 0.4 * (0.5 - rnd.NextDouble());
                            deltaItemValue[n, w] = dw0;
                        }
                    }
                }
            }
        }
        private static Dictionary<NetworkLayer, double[,]> Create2DLayers(int[] networkLayers)
        {
            return new Dictionary<NetworkLayer, double[,]>
            {
                { NetworkLayer.Input, null },
                { NetworkLayer.Hidden, new double[networkLayers[(int)NetworkLayer.Hidden], networkLayers[(int)NetworkLayer.Input]+1] },
                { NetworkLayer.Output, new double[networkLayers[(int)NetworkLayer.Output], networkLayers[(int)NetworkLayer.Hidden]+1] }
            };
        }
        private static Dictionary<NetworkLayer, double[]> Create1DLayers(int[] networkLayers)
        {
            return new Dictionary<NetworkLayer, double[]>
            {
                { NetworkLayer.Input, null },
                { NetworkLayer.Hidden, new double[networkLayers[(int)NetworkLayer.Hidden]] },
                { NetworkLayer.Output, new double[networkLayers[(int)NetworkLayer.Output]] }
            };
        }
        private static Dictionary<NetworkLayer, double[]> CreateFull1DLayers(int[] networkLayers)
        {
            return new Dictionary<NetworkLayer, double[]>
            {
                { NetworkLayer.Input, new double[networkLayers[(int)NetworkLayer.Input]] },
                { NetworkLayer.Hidden, new double[networkLayers[(int)NetworkLayer.Hidden]] },
                { NetworkLayer.Output, new double[networkLayers[(int)NetworkLayer.Output]] }
            };
        }
    }
}

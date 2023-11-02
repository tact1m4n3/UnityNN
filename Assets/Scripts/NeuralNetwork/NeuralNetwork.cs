using System;
using System.Collections.Generic;
using UnityEngine;

public class NeuralNetwork
{
    public LossFn lossFn;
    public List<Layer> layers;

    public NeuralNetwork(LossFn _lossFn)
    {
        lossFn = _lossFn;
        layers = new List<Layer>();
    }

    public void Add(Layer layer)
    {
        layers.Add(layer);
    }

    public float[] Forward(float[] inputs)
    {
        if (layers.Count <= 0) return inputs;

        float[] output = layers[0].Forward(inputs);
        for (int i = 1; i < layers.Count; i++)
            output = layers[i].Forward(output);
        return output;
    }

    public float[] Backward(float[] output, float[] expected, float learnRate)
    {
        float[] doutput = lossFn.Backward(output, expected);
        for (int i = layers.Count - 1; i >= 0; i--)
            doutput = layers[i].Backward(doutput, learnRate);
        return doutput;
    }

    public void Train(float[][] x, float[][] y, int epochsCount, float learnRate)
    {
        int samples = x.Length;

        for (int i = 1; i <= epochsCount; i++)
        {
            float loss = 0;
            for (int j = 0; j < samples; j++)
            {
                float[] output = Forward(x[j]);
                loss += lossFn.Forward(output, y[j]);
                Backward(output, y[j], learnRate);
            }

            loss /= samples;
            Debug.Log($"Epoch {i} Loss {loss}");
        }
    }
}

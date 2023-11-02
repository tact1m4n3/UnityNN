using System;
using UnityEngine;
using Random = UnityEngine.Random;

public enum LayerKind
{
    Linear,
    Sigmoid,
    Tanh,
    ReLU,
}

public interface Layer
{
    LayerKind GetLayerKind();
    float[] Forward(float[] inputs);
    float[] Backward(float[] doutput, float learnRate);
}

public class Linear : Layer
{
    public int inputsCount, neuronsCount;
    public float[] weights;
    public float[] biases;
    private float[] inputs;

    public Linear(int _inputsCount, int _neuronsCount)
    {
        inputsCount = _inputsCount;
        neuronsCount = _neuronsCount;

        weights = new float[neuronsCount * inputsCount];
        biases = new float[neuronsCount];

        RandomizeWeightsAndBiases();
    }

    public Linear(int _inputsCount, int _neuronsCount, float[] _weights, float[] _biases)
    {
        inputsCount = _inputsCount;
        neuronsCount = _neuronsCount;
        weights = _weights;
        biases = _biases;
    }

    public LayerKind GetLayerKind() { return LayerKind.Linear; }

    public float[] Forward(float[] _inputs)
    {
        inputs = _inputs;

        float[] output = new float[neuronsCount];
        for (int i = 0; i < neuronsCount; i++)
        {
            for (int j = 0; j < inputsCount; j++)
                output[i] += inputs[j] * weights[i * inputsCount + j];
            output[i] += biases[i];
        }
        return output;
    }

    public float[] Backward(float[] doutput, float learnRate)
    {
        float[] dinputs = new float[inputsCount];
        for (int i = 0; i < inputsCount; i++)
            for (int j = 0; j < neuronsCount; j++)
                dinputs[i] += doutput[j] * weights[j * inputsCount + i];

        for (int i = 0; i < neuronsCount; i++)
        {
            for (int j = 0; j < inputsCount; j++)
                weights[i * inputsCount + j] -= learnRate * doutput[i] * inputs[j];
            biases[i] -= learnRate * doutput[i];
        }

        return dinputs;
    }

    private void RandomizeWeightsAndBiases()
    {
        for (int i = 0; i < neuronsCount; i++)
        {
            for (int j = 0; j < inputsCount; j++)
                weights[i * inputsCount + j] = Random.Range(-0.5f, 0.5f);
            biases[i] = 0;
        }
    }
}

public class Sigmoid : Layer
{
    float[] output;

    public LayerKind GetLayerKind() { return LayerKind.Sigmoid; }

    public float[] Forward(float[] inputs)
    {
        output = new float[inputs.Length];
        for (int i = 0; i < output.Length; i++)
            output[i] = 1 / (1 + Mathf.Exp(-inputs[i]));
        return output;
    }

    public float[] Backward(float[] doutput, float learnRate)
    {
        float[] dinputs = new float[doutput.Length];
        for (int i = 0; i < dinputs.Length; i++)
            dinputs[i] = doutput[i] * output[i] * (1 - output[i]);
        return dinputs;
    }
}

public class Tanh : Layer
{
    float[] output;

    public LayerKind GetLayerKind() { return LayerKind.Tanh; }

    public float[] Forward(float[] inputs)
    {
        output = new float[inputs.Length];
        for (int i = 0; i < output.Length; i++)
            output[i] = MathF.Tanh(inputs[i]);
        return output;
    }

    public float[] Backward(float[] doutput, float learnRate)
    {
        float[] dinputs = new float[doutput.Length];
        for (int i = 0; i < dinputs.Length; i++)
            dinputs[i] = doutput[i] * (1 - Mathf.Pow(output[i], 2));
        return dinputs;
    }
}

public class ReLU : Layer
{
    private float[] inputs;

    public LayerKind GetLayerKind() { return LayerKind.ReLU; }

    public float[] Forward(float[] _inputs)
    {
        inputs = _inputs;
        float[] output = new float[inputs.Length];
        for (int i = 0; i < output.Length; i++)
            output[i] = (inputs[i] >= 0) ? inputs[i] : 0;
        return output;
    }

    public float[] Backward(float[] doutput, float learnRate)
    {
        float[] dinputs = new float[doutput.Length];
        for (int i = 0; i < dinputs.Length; i++)
            dinputs[i] = (inputs[i] >= 0) ? doutput[i] : 0;
        return dinputs;
    }
}

using UnityEngine;

public enum LossFnKind
{
    MSE,
}

public interface LossFn
{
    LossFnKind GetLossFnType();
    float Forward(float[] output, float[] expected);
    float[] Backward(float[] output, float[] expected);
}

public class MSE : LossFn
{
    public LossFnKind GetLossFnType() { return LossFnKind.MSE; }

    public float Forward(float[] output, float[] expected)
    {
        float loss = 0;
        for (int i = 0; i < output.Length; i++)
            loss += Mathf.Pow(expected[i] - output[i], 2);
        return loss;
    }

    public float[] Backward(float[] output, float[] expected)
    {
        float[] dinputs = new float[output.Length];
        for (int i = 0; i < dinputs.Length; i++)
            dinputs[i] = 2 * (output[i] - expected[i]) / expected.Length;
        return dinputs;
    }
}

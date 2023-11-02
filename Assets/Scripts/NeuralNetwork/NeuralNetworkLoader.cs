using System;
using System.IO;
using System.Collections.Generic;
using UnityEngine;

[Serializable]
public struct LayerData
{
    [SerializeField] public LayerKind kind;
    [SerializeField] public int inputsCount, neuronsCount;
    [SerializeField] public float[] weights;
    [SerializeField] public float[] biases;
}

[Serializable]
public struct NeuralNetworkData
{
    [SerializeField] public LossFnKind lossFnKind;
    [SerializeField] public List<LayerData> layers;
}

public static class NeuralNetworkLoader
{
    public static NeuralNetwork Load(string name)
    {
        string json = File.ReadAllText($"{Application.dataPath}/Networks/{name}.json");
        NeuralNetworkData networkData = JsonUtility.FromJson<NeuralNetworkData>(json);

        LossFn lossFn = null;
        switch (networkData.lossFnKind)
        {
        case LossFnKind.MSE:
            lossFn = new MSE();
            break;
        }

        NeuralNetwork network = new NeuralNetwork(lossFn);

        for (int i = 0; i < networkData.layers.Count; i++)
        {
            LayerData layerData = networkData.layers[i];

            Layer layer = null;
            switch (layerData.kind)
            {
            case LayerKind.Linear:
                layer = new Linear(layerData.inputsCount, layerData.neuronsCount, layerData.weights, layerData.biases);
                break;
            case LayerKind.Sigmoid:
                layer = new Sigmoid();
                break;
            case LayerKind.Tanh:
                layer = new Tanh();
                break;
            case LayerKind.ReLU:
                layer = new ReLU();
                break;
            }

            network.Add(layer);
        }

        return network;
    }

    public static void Save(NeuralNetwork network, string name)
    {
        NeuralNetworkData networkData = new NeuralNetworkData();
        networkData.lossFnKind = network.lossFn.GetLossFnType();

        networkData.layers = new List<LayerData>();
        for (int i = 0; i < network.layers.Count; i++)
        {
            Layer layer = network.layers[i];

            LayerData layerData = new LayerData();
            layerData.kind = layer.GetLayerKind();

            if (layer.GetLayerKind() == LayerKind.Linear)
            {
                Linear linearLayer = (Linear)layer;
                layerData.inputsCount = linearLayer.inputsCount;
                layerData.neuronsCount = linearLayer.neuronsCount;
                layerData.weights = linearLayer.weights;
                layerData.biases = linearLayer.biases;
            }

            networkData.layers.Add(layerData);
        }

        string json = JsonUtility.ToJson(networkData);

        Directory.CreateDirectory($"{Application.dataPath}/Networks");
        File.WriteAllText($"{Application.dataPath}/Networks/{name}.json", json);
    }
}

using System.Threading;
using UnityEngine;
using UnityEditor;

public class MNIST_Trainer : EditorWindow
{
    private MNIST_Dataset mnistDataset;
    private NeuralNetwork network;
    private bool training;
    private Thread thread;
    private int epochsCount;
    private float learnRate;
    private string saveName;

    [MenuItem("Tools/MNIST Trainer")]
    public static void Init()
    {
        GetWindow(typeof(MNIST_Trainer));
    }

    private void OnEnable()
    {
        mnistDataset = MNIST_Dataset.Load();

        ResetNeuralNetwork();
    }

    private void OnDisable()
    {
        if (training)
            thread.Abort();
    }

    private void OnGUI()
    {
        epochsCount = EditorGUILayout.IntField("Epochs Count", epochsCount);
        learnRate = EditorGUILayout.FloatField("Learn Rate", (float)learnRate);

        if (GUILayout.Button("Train") && !training)
        {
            thread = new Thread(Train);
            thread.Start();
            training = true;
        }

        if (GUILayout.Button("Stop Training") && training)
        {
            thread.Abort();
            training = false;
        }

        if (GUILayout.Button("Reset"))
            ResetNeuralNetwork();

        if (training)
            GUILayout.Label("Status: Training");
        else
            GUILayout.Label("Status: Not Training");

        EditorGUILayout.Separator();

        saveName = EditorGUILayout.TextField("Name", saveName);
        if (GUILayout.Button("Save") && !training)
            NeuralNetworkLoader.Save(network, saveName);
        if (GUILayout.Button("Load") && !training)
            network = NeuralNetworkLoader.Load(saveName);
    }

    private void ResetNeuralNetwork()
    {
        network = new NeuralNetwork(new MSE());
        network.Add(new Linear(28 * 28, 20));
        network.Add(new ReLU());
        network.Add(new Linear(20, 20));
        network.Add(new ReLU());
        network.Add(new Linear(20, 10));
        network.Add(new Sigmoid());
    }

    private void Train()
    {
        network.Train(mnistDataset.trainX, mnistDataset.trainY, epochsCount, learnRate);
        training = false;
    }
}

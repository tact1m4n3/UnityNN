using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class MNIST_Tester : MonoBehaviour
{
    [SerializeField] private string networkName;
    [SerializeField] private Image previewImage;
    [SerializeField] private TMP_Text targetText;
    [SerializeField] private TMP_Text predictionText;
    private MNIST_Dataset mnistDataset;
    private NeuralNetwork network;
    private Texture2D texture;
    private byte[] textureData;

    private void Start()
    {
        mnistDataset = MNIST_Dataset.Load();

        network = NeuralNetworkLoader.Load(networkName);

        texture = new Texture2D(mnistDataset.imageWidth, mnistDataset.imageHeight, TextureFormat.Alpha8, false);
        textureData = new byte[mnistDataset.imageWidth * mnistDataset.imageHeight];

        previewImage.sprite = Sprite.Create(texture, new Rect(0.0f, 0.0f, texture.width, texture.height), new Vector2(0.5f, 0.5f), 100.0f);

        NextImage();
    }

    public void NextImage()
    {
        int idx = Random.Range(0, mnistDataset.testSamples);

        float[] inputs = mnistDataset.testX[idx];

        for (int i = 0; i < mnistDataset.imageWidth * mnistDataset.imageHeight; i++)
            textureData[i] = (byte)(255.0 * inputs[i]);

        texture.SetPixelData(textureData, 0);
        texture.Apply();

        float[] output = network.Forward(inputs);
        float[] expected = mnistDataset.testY[idx];

        targetText.text = $"Target: {MaxIdx(expected)}";
        predictionText.text = $"Prediction: {MaxIdx(output)}";
    }

    private int MaxIdx(float[] x)
    {
        int idx = 0;
        float max = float.MinValue;
        for (int i = 0; i < x.Length; i++)
        {
            if (max < x[i])
            {
                idx = i;
                max = x[i];
            }
        }
        return idx;
    }
}

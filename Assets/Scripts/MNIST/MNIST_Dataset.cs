using System.Threading.Tasks;
using UnityEngine;

public class MNIST_Dataset
{
    private static MNIST_Dataset instance;
    public int imageWidth, imageHeight;
    public int trainSamples;
    public float[][] trainX;
    public float[][] trainY;
    public int testSamples;
    public float[][] testX;
    public float[][] testY;

    public static MNIST_Dataset Load()
    {
        if (instance != null)
            return instance;

        MNIST_Dataset mnist = new MNIST_Dataset();

        {
            TextAsset asset = Resources.Load<TextAsset>("Datasets/MNIST/TrainImages");
            byte[] data = asset.bytes;

            int magic = ToInt(data[0..4]);
            int count = ToInt(data[4..8]);
            int height = ToInt(data[8..12]);
            int width = ToInt(data[12..16]);
            data = data[16..];

            float[][] x = new float[count][];
            Parallel.For(0, count, (int i) =>
            {
                x[i] = new float[width * height];
                for (int j = 0; j < width * height; j++)
                    x[i][j] = (float)data[i * width * height + j] * 0.99f / 255 + 0.01f;
            });

            mnist.imageWidth = width;
            mnist.imageHeight = height;
            mnist.trainSamples = count;
            mnist.trainX = x;
        }

        {
            TextAsset asset = Resources.Load<TextAsset>("Datasets/MNIST/TrainLabels");
            byte[] data = asset.bytes;

            int magic = ToInt(data[0..4]);
            int count = ToInt(data[4..8]);
            data = data[8..];

            float[][] y = new float[count][];
            Parallel.For(0, count, (int i) =>
            {
                y[i] = new float[10];
                y[i][data[i]] = 1;
            });

            mnist.trainY = y;
        }

        {
            TextAsset asset = Resources.Load<TextAsset>("Datasets/MNIST/TestImages");
            byte[] data = asset.bytes;

            int magic = ToInt(data[0..4]);
            int count = ToInt(data[4..8]);
            int height = ToInt(data[8..12]);
            int width = ToInt(data[12..16]);
            data = data[16..];

            float[][] x = new float[count][];
            Parallel.For(0, count, (int i) =>
            {
                x[i] = new float[width * height];
                for (int j = 0; j < width * height; j++)
                    x[i][j] = (float)data[i * width * height + j] * 0.99f / 255 + 0.01f;
            });

            mnist.testSamples = count;
            mnist.testX = x;
        }

        {
            TextAsset asset = Resources.Load<TextAsset>("Datasets/MNIST/TestLabels");
            byte[] data = asset.bytes;

            int magic = ToInt(data[0..4]);
            int count = ToInt(data[4..8]);
            data = data[8..];

            float[][] y = new float[count][];
            Parallel.For(0, count, (int i) =>
            {
                y[i] = new float[10];
                y[i][data[i]] = 1;
            });

            mnist.testY = y;
        }

        instance = mnist;
        return mnist;
    }

    private static int ToInt(byte[] arr)
    {
        return (arr[0] << 24) | (arr[1] << 16) | (arr[2] << 8) | arr[3];
    }
}

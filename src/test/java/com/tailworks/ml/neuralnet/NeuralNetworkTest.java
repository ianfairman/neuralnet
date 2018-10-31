package com.tailworks.ml.neuralnet;

import com.tailworks.ml.neuralnet.math.Vec;
import org.junit.Test;

import static java.lang.System.arraycopy;
import static org.junit.Assert.assertEquals;

public class NeuralNetworkTest {

    // Based on forward pass here
    // https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c
    @Test
    public void testEvaluate() {

        double[][][] initWeights = {
                {{0.1, 0.3, 0.4}, {0.2, 0.2, 0.3}, {0.3, 0.7, 0.9}},
                {{0.2, 0.3, 0.6}, {0.3, 0.5, 0.4}, {0.5, 0.7, 0.8}},
                {{0.1, 0.3, 0.5}, {0.4, 0.7, 0.2}, {0.8, 0.2, 0.9}},
        };

        NeuralNetwork network =
                new NeuralNetwork.Builder(3)
                        .addLayer(new Layer(3, Activation.ReLU, 1))
                        .addLayer(new Layer(3, Activation.LogSigmoid, 1))
                        .addLayer(new Layer(3, Activation.Softmax_broken, 1))
                        .initWeights((weights, layer) -> {
                            double[][] data = weights.getData();
                            for (int row = 0; row < data.length; row++)
                                arraycopy(initWeights[layer][row], 0, data[row], 0, data[0].length);
                        })
                        .create();

        Vec out = network.evaluate(new Vec(0.1, 0.2, 0.7)).getOutput();

        assertEquals(0.26979, out.getData()[0], 0.1);
        assertEquals(0.32227, out.getData()[1], 0.1);
        assertEquals(0.40793, out.getData()[2], 0.1);
    }


    // Based on forward pass here
    // https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    @Test
    public void testEvaluateAndLearn() {
        double[][][] initWeights = {
                {{0.15, 0.20}, {0.25, 0.30}},
                {{0.40, 0.45}, {0.50, 0.55}},
        };

        NeuralNetwork network =
                new NeuralNetwork.Builder(2)
                        .addLayer(new Layer(2, Activation.LogSigmoid, new Vec(0.35, 0.35)))
                        .addLayer(new Layer(2, Activation.LogSigmoid, new Vec(0.60, 0.60)))
                        .setLearningRate(0.5)
                        .initWeights((weights, layer) -> {
                            double[][] data = weights.getData();
                            for (int row = 0; row < data.length; row++)
                                arraycopy(initWeights[layer][row], 0, data[row], 0, data[0].length);
                        })
                        .create();


        double eps = 0.00000001;

        Vec wanted = new Vec(0.01, 0.99);
        Vec input = new Vec(0.05, 0.1);

        Result result = network.evaluate(input, wanted);

        Vec out = result.getOutput();

        assertEquals(0.298371109, result.getCost(), eps);
        assertEquals(0.75136507, out.getData()[0], eps);
        assertEquals(0.77292846, out.getData()[1], eps);

        network.learn(wanted);

        result = network.evaluate(input, wanted);
        out = result.getOutput();

        assertEquals(0.28047144, result.getCost(), eps);
        assertEquals(0.72844176, out.getData()[0], eps);
        assertEquals(0.77837692, out.getData()[1], eps);

        for (int i = 0; i < 10000 - 2; i++) {
            network.learn(wanted);
            result = network.evaluate(input, wanted);
        }

        out = result.getOutput();
        assertEquals(0.0000024485, result.getCost(), eps);
        assertEquals(0.011587777, out.getData()[0], eps);
        assertEquals(0.9884586899, out.getData()[1], eps);
    }

    @Test
    public void testEvaluateAndLearn2() {

        NeuralNetwork network =
                new NeuralNetwork.Builder(4)
                        .addLayer(new Layer(6, Activation.LogSigmoid, 0.5))
                        .addLayer(new Layer(14, Activation.LogSigmoid, 0.5))
                        .setLearningRate(0.5)
                        .initWeights(new WeightInitializer.Random(-0.5, 0.5))
                        .create();


        int trainInputs[][] = new int[][]{
                {1, 1, 1, 0},
                {1, 1, 0, 0},
                {0, 1, 1, 0},
                {1, 0, 1, 0},
                {1, 0, 0, 0},
                {0, 1, 0, 0},
                {0, 0, 1, 0},
                {1, 1, 1, 1},
                {1, 1, 0, 1},
                {0, 1, 1, 1},
                {1, 0, 1, 1},
                {1, 0, 0, 1},
                {0, 1, 0, 1},
                {0, 0, 1, 1}
        };

        int trainOutput[][] = new int[][]{
                {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}
        };


        int cnt = 0;
        for (int i = 0; i < 4000; i++) {
            Vec input = new Vec(trainInputs[cnt]);
            Vec expected = new Vec(trainOutput[cnt]);
            network.evaluate(input);
            network.learn(expected);
            cnt = (cnt + 1) % trainInputs.length;
        }

        for (int i = 0; i < trainInputs.length; i++) {
            Result result = network.evaluate(new Vec(trainInputs[i]));
            int ix = result.getOutput().indexOfLargestElement();
            assertEquals(new Vec(trainOutput[i]), new Vec(trainOutput[ix]));
        }
    }
}

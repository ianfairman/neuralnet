package com.tailworks.ml.neuralnet;

import com.tailworks.ml.neuralnet.math.Vec;
import org.junit.Test;

import static com.tailworks.ml.neuralnet.Activation.*;
import static java.lang.System.arraycopy;
import static org.junit.Assert.*;

public class NeuralNetworkTest {

    private static final double EPS = 0.00001;

    @Test
    public void testEvaluate() {

        double[][][] initWeights = {
                {{0.1, 0.2, 0.3}, {0.3, 0.2, 0.7}, {0.4, 0.3, 0.9}},
                {{0.2, 0.3, 0.5}, {0.3, 0.5, 0.7}, {0.6, 0.4, 0.8}},
                {{0.1, 0.4, 0.8}, {0.3, 0.7, 0.2}, {0.5, 0.2, 0.9}}
        };

        NeuralNetwork network =
                new NeuralNetwork.Builder(3)
                        .addLayer(new Layer(3, ReLU, 1))
                        .addLayer(new Layer(3, Sigmoid, 1))
                        .addLayer(new Layer(3, Softmax, 1))
                        .initWeights((weights, layer) -> {
                            double[][] data = weights.getData();
                            for (int row = 0; row < data.length; row++)
                                arraycopy(initWeights[layer][row], 0, data[row], 0, data[0].length);
                        })
                        .create();

        Vec out = network.evaluate(new Vec(0.1, 0.2, 0.7)).getOutput();

        double[] data = out.getData();
        assertEquals(0.198446894232731, data[0], EPS);
        assertEquals(0.2853555304250049, data[1], EPS);
        assertEquals(0.5161975753422642, data[2], EPS);
        assertEquals(1, data[0] + data[1] + data[2], EPS);
    }


    // Based on forward pass here
    // https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    @Test
    public void testEvaluateAndLearn() {
        double[][][] initWeights = {
                {{0.15, 0.25}, {0.20, 0.30}},
                {{0.40, 0.50}, {0.45, 0.55}},
        };

        NeuralNetwork network =
                new NeuralNetwork.Builder(2)
                        .addLayer(new Layer(2, Sigmoid, new Vec(0.35, 0.35)))
                        .addLayer(new Layer(2, Sigmoid, new Vec(0.60, 0.60)))
                        .setCostFunction(new CostFunction.L2Half())
                        .setLearningRate(0.5)
                        .initWeights((weights, layer) -> {
                            double[][] data = weights.getData();
                            for (int row = 0; row < data.length; row++)
                                arraycopy(initWeights[layer][row], 0, data[row], 0, data[0].length);
                        })
                        .create();


        Vec wanted = new Vec(0.01, 0.99);
        Vec input = new Vec(0.05, 0.1);

        Result result = network.evaluate(input, wanted, true);

        Vec out = result.getOutput();

        assertEquals(0.29837110, result.getCost(), EPS);
        assertEquals(0.75136507, out.getData()[0], EPS);
        assertEquals(0.77292846, out.getData()[1], EPS);

        network.updateFromLearning();

        result = network.evaluate(input, wanted, true);
        out = result.getOutput();

        assertEquals(0.28047144, result.getCost(), EPS);
        assertEquals(0.72844176, out.getData()[0], EPS);
        assertEquals(0.77837692, out.getData()[1], EPS);

        for (int i = 0; i < 10000 - 2; i++) {
            network.updateFromLearning();
            result = network.evaluate(input, wanted, true);
        }

        out = result.getOutput();
        assertEquals(0.0000024485, result.getCost(), EPS);
        assertEquals(0.011587777, out.getData()[0], EPS);
        assertEquals(0.9884586899, out.getData()[1], EPS);
    }

    @Test
    public void testEvaluateAndLearn2() {

        NeuralNetwork network =
                new NeuralNetwork.Builder(4)
                        .addLayer(new Layer(6, Sigmoid, 0.5))
                        .addLayer(new Layer(14, Sigmoid, 0.5))
                        .setCostFunction(new CostFunction.L2())
                        .setLearningRate(1)
                        .initWeights(new Initializer.XavierNormal())
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
        for (int i = 0; i < 1100; i++) {
            Vec input = new Vec(trainInputs[cnt]);
            Vec expected = new Vec(trainOutput[cnt]);
            network.evaluate(input, expected, true);
            network.updateFromLearning();
            cnt = (cnt + 1) % trainInputs.length;
        }

        for (int i = 0; i < trainInputs.length; i++) {
            Result result = network.evaluate(new Vec(trainInputs[i]));
            int ix = result.getOutput().indexOfLargestElement();
            assertEquals(new Vec(trainOutput[i]), new Vec(trainOutput[ix]));
        }
    }

    @Test
    public void testBatching() {

        NeuralNetwork n1 =
                new NeuralNetwork.Builder(2)
                        .addLayer(new Layer(2, ReLU, 0.5))
                        .addLayer(new Layer(2, Sigmoid, 0.5))
                        .initWeights(new Initializer.XavierNormal())
                        .create();

        Vec i1 = new Vec(0.1, 0.7);
        Vec i2 = new Vec(-0.2, 0.3);
        Vec w1 = new Vec(0.2, 0.1);
        Vec w2 = new Vec(1.2, -0.4);

        // First, evaluate without learning
        Result out1a = n1.evaluate(i1, w1, false);
        Result out2a = n1.evaluate(i2, w2, false);

        // Then this should do nothing to the net
        n1.updateFromLearning();

        Result out1b = n1.evaluate(i1, w1, false);
        Result out2b = n1.evaluate(i2, w2, false);
        assertEquals(out1a.getOutput(), out1b.getOutput());
        assertEquals(out2a.getOutput(), out2b.getOutput());

        // Now, evaluate without learning
        out1a = n1.evaluate(i1, w1, true);
        out2a = n1.evaluate(i2, w2, true);

        double cost1BeforeLearning = out1b.getCost();
        double cost2BeforeLearning = out2b.getCost();

        // First verify that we still have not changed any weights
        assertEquals(out1a.getOutput(), out1b.getOutput());
        assertEquals(out2a.getOutput(), out2b.getOutput());

        // This should however change things ...
        n1.updateFromLearning();

        out1b = n1.evaluate(i1, w1, true);
        out2b = n1.evaluate(i2, w2, true);

        assertNotEquals(out1a.getOutput(), out1b.getOutput());
        assertNotEquals(out2a.getOutput(), out2b.getOutput());

        // ... and the cost should be lower
        double cost1AfterLearning = out1b.getCost();
        double cost2AfterLearning = out2b.getCost();

        assertTrue(cost1AfterLearning < cost1BeforeLearning);
        assertTrue(cost2AfterLearning < cost2BeforeLearning);

    }

}

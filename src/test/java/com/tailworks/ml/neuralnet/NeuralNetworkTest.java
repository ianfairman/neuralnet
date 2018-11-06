package com.tailworks.ml.neuralnet;

import com.tailworks.ml.neuralnet.math.Vec;
import org.junit.Test;

import static com.tailworks.ml.neuralnet.Activation.*;
import static java.lang.System.arraycopy;
import static org.junit.Assert.assertEquals;

public class NeuralNetworkTest {

    public static final double EPS = 0.00001;

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
                        .addLayer(new Layer(3, LogSigmoid, 1))
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
                        .addLayer(new Layer(2, LogSigmoid, new Vec(0.35, 0.35)))
                        .addLayer(new Layer(2, LogSigmoid, new Vec(0.60, 0.60)))
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

        Result result = network.evaluate(input, wanted);

        Vec out = result.getOutput();

        assertEquals(0.29837110, result.getCost(), EPS);
        assertEquals(0.75136507, out.getData()[0], EPS);
        assertEquals(0.77292846, out.getData()[1], EPS);

        network.learn(wanted);

        result = network.evaluate(input, wanted);
        out = result.getOutput();

        assertEquals(0.28047144, result.getCost(), EPS);
        assertEquals(0.72844176, out.getData()[0], EPS);
        assertEquals(0.77837692, out.getData()[1], EPS);

        for (int i = 0; i < 10000 - 2; i++) {
            network.learn(wanted);
            result = network.evaluate(input, wanted);
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
                        .addLayer(new Layer(6, LogSigmoid, 0.5))
                        .addLayer(new Layer(14, LogSigmoid, 0.5))
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
        for (int i = 0; i < 20000; i++) {
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

    @Test
    public void testDeterminism() {

        NeuralNetwork n1 =
                new NeuralNetwork.Builder(2)
                        .addLayer(new Layer(2, ReLU, 0.5))
                        .addLayer(new Layer(2, LogSigmoid, 0.5))
                        .initWeights(new Initializer.XavierNormal())
                        .create();

        NeuralNetwork n2 =
                new NeuralNetwork.Builder(2)
                        .addLayer(new Layer(2, ReLU, 0.5))
                        .addLayer(new Layer(2, LogSigmoid, 0.5))
                        .initWeights(new Initializer.XavierNormal())
                        .create();

        Vec in = new Vec(0.1, 0.7);

        Vec o1a = n1.evaluate(in).getOutput();
        Vec o1b = n1.evaluate(in).getOutput();
        n1.evaluate(in);
        n1.evaluate(in);
        n1.evaluate(in);
        n1.evaluate(in);
        Vec o2 = n2.evaluate(in).getOutput();

        double eps = 0.00000001;

        assertEquals(o1a.getData()[0], o1b.getData()[0], eps);
        assertEquals(o1a.getData()[1], o1b.getData()[1], eps);

        assertEquals(o1a.getData()[0], o2.getData()[0], eps);
        assertEquals(o1a.getData()[1], o2.getData()[1], eps);

        Vec wanted = new Vec(0.2, 0.1);
        n1.learn(wanted);
        n2.learn(wanted);

        o1a = n1.evaluate(in).getOutput();
        o2 = n2.evaluate(in).getOutput();

        assertEquals(o1a.getData()[0], o2.getData()[0], eps);
        assertEquals(o1a.getData()[1], o2.getData()[1], eps);

        n1.learn(wanted);
        n2.learn(wanted);
        n1.learn(wanted);
        n2.learn(wanted);

        o1a = n1.evaluate(in).getOutput();
        o2 = n2.evaluate(in).getOutput();

        assertEquals(o1a.getData()[0], o2.getData()[0], eps);
        assertEquals(o1a.getData()[1], o2.getData()[1], eps);

    }

}
